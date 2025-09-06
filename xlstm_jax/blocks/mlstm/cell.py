# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from ...components.init import bias_linspace_initializer
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_stabilized_simple, recurrent_step_stabilized_simple


@dataclass(unsafe_hash=True, order=True)
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nnx.Module):
    def __init__(
        self,
        config: mLSTMCellConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        # Store configuration parameters for easier access
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.dim_per_head = self.embedding_dim // self.num_heads

        # Set backend functions
        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        Linear = partial(
            nnx.Linear,
            in_features=3 * config.embedding_dim,
            out_features=config.num_heads,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        # Gate projections
        self.igate = Linear(
            bias_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.1, dtype=dtype),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.fgate = Linear(
            bias_init=nnx.with_partitioning(
                bias_linspace_initializer(start=3.0, end=6.0),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        # Output normalization
        self.outnorm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=True,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.causal_mask = nnx.Variable(
            jnp.tril(
                jnp.ones(
                    (config.context_length, config.context_length),
                    dtype=jnp.bool,
                )
            )
        )

    def _split_heads(self, el: jax.Array):
        return rearrange(el, "b s (nh hd) -> b nh s hd", nh=self.num_heads)

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = jnp.concatenate([q, k, v], axis=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = jnp.transpose(igate_preact, (0, 2, 1))
        igate_preact = igate_preact[..., None]  # (B, NH, S, 1)

        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = jnp.transpose(fgate_preact, (0, 2, 1))
        fgate_preact = fgate_preact[..., None]  # (B, NH, S, 1)

        # slice a causal mask of appropriate size to avoid creating
        # a new array temporary every time in the backend fn
        causal_mask = self.causal_mask.value[:S, :S]

        # Apply mLSTM backend function
        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        # Apply normalization
        h_state_norm = self.outnorm(h_state)  # (B, S, D)

        return h_state_norm
