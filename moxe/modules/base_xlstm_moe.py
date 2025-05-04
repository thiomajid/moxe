import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import lax

from moxe.config import MoxEConfig
from xlstm_jax.xlstm_block_stack import xLSTMBlockStack


class xLSTMMoELayer(nnx.Module):
    gate: nnx.Linear
    experts: list[nnx.Module]
    sequence_mixer: xLSTMBlockStack

    def __init__(self, config: MoxEConfig) -> None:
        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts

    # code adapted from transformers.models.mixtral.modeling_mixtral.py
    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.sequence_mixer(hidden_states)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)  # (B*S, D)

        router_logits = self.gate(flat_hidden_states)  # (B*S, num_experts)
        routing_weights = jax.nn.softmax(router_logits, axis=1)
        routing_weights, selected_experts = lax.top_k(routing_weights, self.top_k)

        # normalize routing weights to sum up to 1
        routing_weights = routing_weights / routing_weights.sum(axis=-1, keepdims=True)
        routing_weights = jnp.astype(routing_weights, hidden_states.dtype)

        # for each expert find tokens that are routed to it
        final_hidden_states = jnp.zeros_like(flat_hidden_states)
        expert_mask = rearrange(
            jax.nn.one_hot(selected_experts, num_classes=self.num_experts),
            "batch top_k experts -> experts top_k batch",
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # find tokens that are routed to the expert
            idx, top_x = jnp.where(expert_mask[expert_idx])

            if top_x.size == 0:  # Skip unused experts
                continue

            if top_x.max() >= flat_hidden_states.size(0):
                raise IndexError("Index out of bounds in top_x")

            # Reshape to 3D for mLSTMBlock
            current_state = flat_hidden_states[top_x]  # (num_selected, D)
            current_state = jnp.expand_dims(
                current_state, axis=1
            )  # (num_selected, 1, D)

            expert_output = expert_layer(current_state)  # (num_selected, 1, D)
            expert_output = jnp.squeeze(expert_output, axis=1)  # (num_selected, D)

            weighted_output = expert_output * routing_weights[top_x, idx, None]
            weighted_output = expert_output * jnp.expand_dims(
                routing_weights[top_x, idx], axis=-1
            )

            # Accumulate results
            final_hidden_states = final_hidden_states.at[top_x].add(weighted_output)

        # reshape back to 3D
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
