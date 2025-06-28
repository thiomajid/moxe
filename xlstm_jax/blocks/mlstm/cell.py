# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.init import bias_linspace_initializer
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_stabilized_simple, recurrent_step_stabilized_simple


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nnx.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        # Store configuration parameters for easier access
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.dim_per_head = self.embedding_dim // self.num_heads

        # Set backend functions
        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        # Gate projections
        self.igate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.normal(stddev=0.1, dtype=dtype),
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.fgate = nnx.Linear(
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=bias_linspace_initializer(start=3.0, end=6.0),
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=True,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
        )

        # Create causal mask buffer
        # INFO: if the mask is not an instance of nnx.Param, the jit compilation
        # fails when nnx will try to flatten the mLSTMCell's pytree
        self.causal_mask = nnx.Variable(
            jnp.tril(
                jnp.ones((config.context_length, config.context_length), dtype=bool)
            )
        )

    # @nnx.jit
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        """Forward pass for parallel processing of the entire sequence.

        Args:
            q: Query tensor of shape (B, S, H)
            k: Key tensor of shape (B, S, H)
            v: Value tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        B, S, _ = q.shape  # (B, S, H)

        # Combine inputs for gate computation
        if_gate_input = jnp.concatenate([q, k, v], axis=-1)

        # Reshape for multi-head processing
        q_reshaped = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k_reshaped = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v_reshaped = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        # Transpose to put heads dimension before sequence dimension
        q_transposed = jnp.transpose(q_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        k_transposed = jnp.transpose(k_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)
        v_transposed = jnp.transpose(v_reshaped, (0, 2, 1, 3))  # (B, NH, S, DH)

        # Compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = jnp.transpose(igate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, S, 1)

        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = jnp.transpose(fgate_preact, (0, 2, 1))[
            ..., None
        ]  # (B, NH, S, 1)

        # Get causal mask of appropriate size
        causal_mask = self.causal_mask[:S, :S]

        # Apply mLSTM backend function
        h_state = self.backend_fn(
            queries=q_transposed,
            keys=k_transposed,
            values=v_transposed,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        # Apply normalization
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)

        # Reshape to original dimensions
        h_state_norm = jnp.transpose(h_state_norm, (0, 2, 1, 3))  # (B, S, NH, DH)
        h_state_norm = h_state_norm.reshape(B, S, -1)  # (B, S, H)

        return h_state_norm

    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        self.outnorm.reset_parameters(rngs)

        # forget gate initialization
        self.fgate.kernel = nnx.Param(jnp.zeros_like(self.fgate.kernel))

        self.fgate.bias = nnx.Param(
            jnp.linspace(3.0, 6.0, num=self.fgate.bias.value.size).reshape(
                self.fgate.bias.value.shape
            )
        )
        # input gate initialization
        self.igate.kernel = nnx.Param(jnp.zeros_like(self.igate.kernel))
        self.igate.bias = nnx.Param(jnp.zeros_like(self.igate.bias))
