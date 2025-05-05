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

        # Reshape weights for easier indexing within scan body
        num_tokens = batch_size * sequence_length  # Calculate max size statically
        flat_expert_weights = expert_weights.reshape(num_tokens, self.top_k)

        def _scan_body(carry: tuple[jax.Array, int], _):
            state, expert_idx = carry
            # expert_mask shape: (num_experts, top_k, num_tokens)
            current_expert_mask = expert_mask[expert_idx]  # Shape (top_k, num_tokens)

            # Use jnp.where with size and fill_value=-1. Returns tuple of index arrays.
            idx_k, idx_token = jnp.where(
                current_expert_mask, size=num_tokens, fill_value=-1
            )
            # idx_k corresponds to top_k dimension, idx_token corresponds to token dimension
            # Both have shape (num_tokens,) due to the `size` argument.

            # Create mask for valid entries (where token index is not -1)
            valid_indices_mask = idx_token != -1  # Shape (num_tokens,)

            # Use safe indices for gathering (replace -1 with 0 to avoid index errors)
            safe_idx_k = jnp.where(valid_indices_mask, idx_k, 0)
            safe_idx_token = jnp.where(valid_indices_mask, idx_token, 0)

            # Gather states using safe_idx_token
            gathered_states = flat_hidden_states[
                safe_idx_token
            ]  # Shape (num_tokens, hidden_dim)
            # Mask out contributions from padded (-1) indices
            masked_gathered_states = gathered_states * valid_indices_mask[:, None]

            # Reshape for expert input: (batch=num_tokens, seq=1, dim=hidden_dim)
            expert_input = jnp.expand_dims(masked_gathered_states, axis=1)

            # Apply expert via lax.switch
            # Ensure expert functions can handle potentially zeroed inputs gracefully
            expert_output_padded = lax.switch(
                expert_idx, self._expert_branches, expert_input
            )
            # Squeeze sequence dimension: (num_tokens, hidden_dim)
            expert_output_squeezed = jnp.squeeze(expert_output_padded, axis=1)

            # Gather weights using safe_idx_token and safe_idx_k
            # flat_expert_weights shape: (num_tokens, top_k)
            gathered_weights = flat_expert_weights[
                safe_idx_token, safe_idx_k
            ]  # Shape (num_tokens,)
            # Mask out contributions from padded (-1) indices
            masked_weights = gathered_weights * valid_indices_mask

            # Apply weights (element-wise)
            weighted_output = (
                expert_output_squeezed * masked_weights[:, None]
            )  # (num_tokens, hidden_dim)

            # Accumulate results using segment_sum
            # Sum contributions in weighted_output based on the original token index (safe_idx_token)
            update_for_state = jax.ops.segment_sum(
                data=weighted_output,
                segment_ids=safe_idx_token,
                num_segments=num_tokens,  # Total number of segments = num_tokens
                indices_are_sorted=False,  # Indices from where are not sorted
            )

            updated_state = state + update_for_state

            # Return new carry state and None for the scanned output per iteration
            return (updated_state, expert_idx + 1), None

        # Initial state for scan
        initial_carry = (final_hidden_states, 0)

        # Run the scan over the number of experts
        final_carry, _ = lax.scan(
            _scan_body,
            init=initial_carry,
            xs=None,  # No input sequence needed for scan itself
            length=self.num_experts,
        )

        # Final hidden states are the first element of the final carry tuple
        final_hidden_states = final_carry[0].reshape(
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
