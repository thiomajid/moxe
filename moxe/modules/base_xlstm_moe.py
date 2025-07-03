import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh

from moxe.config import MoxEConfig
from moxe.modules.gate import StandardMoEGate
from xlstm_jax.xlstm_block_stack import xLSTMBlockStack


class xLSTMMoELayer(nnx.Module):
    experts: list[nnx.Module]
    sequence_mixer: xLSTMBlockStack

    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ) -> None:
        self.top_k = config.top_k_experts
        self.num_experts = config.num_experts

        self.gate = StandardMoEGate(config, mesh=mesh, rngs=rngs, dtype=dtype)

    # code adapted from transformers.models.mixtral.modeling_mixtral.py
    def __call__(self, h_t: jax.Array, *args):
        h_t = self.sequence_mixer(h_t)

        B, S, D = h_t.shape

        router_logits = self.gate(h_t)  # (B*S, num_experts)
        router_probs = jax.nn.softmax(router_logits, axis=1)
        top_k_weights, top_k_indices = lax.top_k(router_probs, self.top_k)

        # normalize routing weights to sum up to 1
        top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
        top_k_weights = jnp.astype(top_k_weights, h_t.dtype)

        flat_h_t = h_t.reshape(B * S, D)

        # Expert-level parallelism: each expert processes all its assigned tokens
        def compute_expert_outputs(expert_idx: int, inputs: jax.Array):
            """Compute outputs for a single expert across all tokens"""
            # Create mask for tokens assigned to this expert
            # top_k_indices: (B*S, top_k), expert_idx: scalar
            expert_mask = top_k_indices == expert_idx  # (B*S, top_k)

            # Get weights for this expert across all tokens
            expert_weights = jnp.where(expert_mask, top_k_weights, 0.0)  # (B*S, top_k)
            expert_weights = jnp.sum(expert_weights, axis=-1)  # (B*S,)

            # Create token selection mask (any token that routes to this expert)
            token_mask = jnp.any(expert_mask, axis=-1)  # (B*S,)

            # Apply expert to ALL tokens (will be masked out later)
            # expert_input = inputs.reshape(B * S, 1, D)
            expert_input = inputs.reshape(B * S, 1, D)
            expert_output: jax.Array = lax.switch(
                expert_idx, self.experts, operand=expert_input
            )
            expert_output = expert_output.reshape(B * S, D)  # (B*S, D)

            # Apply weights and mask
            weighted_output = expert_output * expert_weights[..., None]  # (B*S, D)
            masked_output = weighted_output * token_mask[..., None]  # (B*S, D)

            return masked_output

        # Vectorize across all experts
        expert_outputs = jax.vmap(compute_expert_outputs, in_axes=(0, None))(
            jnp.arange(self.num_experts), flat_h_t
        )  # (num_experts, B*S, D)

        # Sum contributions from all experts
        final_outputs = jnp.sum(expert_outputs, axis=0)  # (B*S, D)
        # reshape back to 3D
        final_hidden_states = final_outputs.reshape(B, S, D)

        return final_hidden_states
