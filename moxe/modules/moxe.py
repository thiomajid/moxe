from copy import deepcopy

import jax
import jax.numpy as jnp
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
        assert config.num_experts > 0, "At least 2 experts per layer"

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
        h_t: jax.Array,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
    ):
        h_t = self.sequence_mixer(h_t)
        B, S, D = h_t.shape
        gate_logits: jax.Array | None = None  # (B*S, E)
        gate_output: ConditionedGateOutput | None = None

        if isinstance(self.gate, BiasConditionedGate):
            gate_output = self.gate(
                h_t,
                compute_d_loss=compute_d_loss,
                compute_group_loss=compute_group_loss,
            )
            gate_logits = gate_output.conditioned_logits
        else:
            flat_h_t = h_t.reshape(-1, D)
            gate_logits = self.gate(flat_h_t)

        router_probs = jax.nn.softmax(gate_logits, axis=1)
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
            # Reshape for xLSTM: (B*S, D) -> (B*S, 1, D)
            # expert_input = inputs.reshape(B * S, 1, D)
            # expert_output = self.experts[expert_idx](inputs)  # (B*S, 1, D)
            expert_output: jax.Array = lax.switch(
                expert_idx, self.experts, operand=inputs
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
        final_hidden_states = final_outputs.reshape(B, S, D)

        # --- Rest of the function remains the same ---
        z_loss = lax.cond(
            self.router_type == SparsityGateType.STANDARD,
            lambda: router_z_loss(gate_logits),
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
            router_logits=gate_logits,
            router_probs=router_probs,
            hidden_states=final_hidden_states,
            conditioned_output=gate_output,
            z_loss=z_loss,
            load_balancing_loss=load_balancing_loss,
            expert_load=expert_load,
            expert_token_counts=expert_token_counts,
        )
