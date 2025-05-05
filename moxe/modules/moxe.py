from copy import deepcopy

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import lax

from xlstm_jax.xlstm_block_stack import xLSTMBlockStack

from ..config import MoxEConfig
from ..ops import auxiliary_load_balancing_loss, router_z_loss
from ..output import ConditionedGateOutput, MoxELayerOutput, SparsityGateType
from ..utils.types import get_expert_modules
from .gate import BiasConditionedGate


class MoxELayer(nnx.Module):
    def __init__(self, config: MoxEConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        assert config.num_experts % 2 == 0, "num_experts must be even"

        self.expert_per_group = config.num_experts // 2
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.gamma = config.gamma
        self.router_type = config.router_type

        mixer_config = deepcopy(config.xlstm)
        mixer_config.num_blocks = 2
        mixer_config.slstm_at = [0]
        _block_map = [1, 0]
        mixer_config._block_map = ",".join(map(str, _block_map))
        self.sequence_mixer = xLSTMBlockStack(mixer_config, rngs=rngs, dtype=dtype)

        self.gate = self.__create_router(config, rngs, dtype=dtype)
        self.experts = get_expert_modules(config, rngs=rngs, dtype=dtype)

        self._expert_branches = [lambda x: expert(x) for expert in self.experts]

    def __create_router(self, config: MoxEConfig, rngs: nnx.Rngs, dtype=jnp.float32):
        match config.router_type:
            case SparsityGateType.CONDITIONED_ADDITION:
                return BiasConditionedGate(config, rngs=rngs, dtype=dtype)
            case SparsityGateType.STANDARD:
                return nnx.Linear(
                    config.xlstm.embedding_dim,
                    self.num_experts,
                    bias=config.gate_bias,
                    dtype=dtype,
                    param_dtype=dtype,
                    rngs=rngs,
                )

        raise ValueError(
            f"Unsupported router type: {config.router_type}"
            f" Supported values are {SparsityGateType.values()}"
        )

    def __call__(
        self,
        h_t: jnp.ndarray,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
    ):
        h_t = self.sequence_mixer(h_t)
        batch_size, sequence_length, hidden_dim = h_t.shape
        router_logits: jnp.ndarray | None = None
        gate_output: ConditionedGateOutput | None = None

        if isinstance(self.gate, BiasConditionedGate):
            gate_output = self.gate(
                h_t,
                compute_d_loss=compute_d_loss,
                compute_group_loss=compute_group_loss,
            )
            router_logits = gate_output.conditioned_logits
        else:
            router_logits = self.gate(h_t)

        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_weights, expert_indices = lax.top_k(router_probs, self.top_k)

        expert_weights = expert_weights / jnp.sum(
            expert_weights, axis=-1, keepdims=True
        )
        expert_weights = expert_weights.astype(h_t.dtype)

        flat_hidden_states = h_t.reshape(-1, hidden_dim)
        final_hidden_states = jnp.zeros_like(flat_hidden_states, dtype=h_t.dtype)
        expert_mask = rearrange(
            jax.nn.one_hot(expert_indices, num_classes=self.num_experts),
            "(b s) top_k experts -> experts top_k (b s)",  # Adjusted rearrange string
            b=batch_size,
            s=sequence_length,
        )

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]

            # find tokens that are routed to the expert
            idx, top_x = jnp.where(expert_mask[expert_idx])

            # Reshape to 3D for mLSTMBlock
            current_state = flat_hidden_states[top_x]  # (num_selected, D)
            current_state = jnp.expand_dims(
                current_state, axis=1
            )  # (num_selected, 1, D)

            expert_output = expert_layer(current_state)  # (num_selected, 1, D)
            expert_output = jnp.squeeze(expert_output, axis=1)  # (num_selected, D)

            weighted_output = expert_output * expert_weights[top_x, idx, None]
            weighted_output = expert_output * jnp.expand_dims(
                expert_weights[top_x, idx], axis=-1
            )

            # Accumulate results
            final_hidden_states = final_hidden_states.at[top_x].add(weighted_output)

        # reshape back to 3D
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        # --- Rest of the function remains the same ---
        z_loss = lax.cond(
            self.router_type == SparsityGateType.STANDARD,
            lambda: router_z_loss(router_logits),
            lambda: gate_output.z_loss,
        )

        load_balancing_loss, expert_load, expert_token_counts = lax.cond(
            self.router_type == SparsityGateType.STANDARD,
            lambda: auxiliary_load_balancing_loss(
                num_experts=self.num_experts,
                router_probs=router_probs,
                top_k=self.top_k,
            ),
            lambda: (
                gate_output.load_balancing_loss,
                gate_output.expert_load,
                gate_output.expert_token_counts,
            ),
        )

        return MoxELayerOutput(
            router_logits=router_logits,
            router_probs=router_probs,
            hidden_states=final_hidden_states,
            conditioned_output=gate_output,
            z_loss=z_loss,
            load_balancing_loss=load_balancing_loss,
            expert_load=expert_load,
            expert_token_counts=expert_token_counts,
        )
