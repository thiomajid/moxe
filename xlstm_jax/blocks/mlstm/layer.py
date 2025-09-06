# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_initializer, wang_initializer
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass(unsafe_hash=True, order=True)
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    _num_blocks: int = 1
    _inner_embedding_dim: Optional[int] = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim


class mLSTMLayer(nnx.Module):
    def __init__(
        self,
        config: mLSTMLayerConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.promote_dtype = dtypes.promote_dtype
        self.dtype = dtype
        self.inner_embedding_dim = config._inner_embedding_dim

        Linear = partial(
            nnx.Linear,
            use_bias=config.bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        # Up-projection
        self.proj_up = Linear(
            in_features=config.embedding_dim,
            out_features=2 * config._inner_embedding_dim,
            kernel_init=nnx.with_partitioning(
                small_init_initializer(dim=config.embedding_dim),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        # QKV projections
        num_proj_heads = round(config._inner_embedding_dim // config.qkv_proj_blocksize)

        qkv_config = LinearHeadwiseExpandConfig(
            in_features=config._inner_embedding_dim,
            num_heads=num_proj_heads,
            bias=config.bias,
        )

        LinearQKV = partial(
            LinearHeadwiseExpand,
            config=qkv_config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializer=nnx.initializers.normal(
                    math.sqrt(2 / 5 / qkv_config.in_features)
                ),
                sharding=(None, None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                initializer=nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.q_proj = LinearQKV()
        self.k_proj = LinearQKV()
        self.v_proj = LinearQKV()

        # Convolutional layer
        self.conv1d = CausalConv1d(
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            config=CausalConv1dConfig(
                feature_dim=config._inner_embedding_dim,
                kernel_size=config.conv1d_kernel_size,
            ),
        )

        # mLSTM cell
        self.mlstm_cell = mLSTMCell(
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            config=mLSTMCellConfig(
                context_length=config.context_length,
                embedding_dim=config._inner_embedding_dim,
                num_heads=config.num_heads,
            ),
        )

        self.ogate_act_fn = jax.nn.swish

        # Learnable skip connection parameter
        self.learnable_skip = nnx.Param(
            jnp.empty(config._inner_embedding_dim, dtype=param_dtype),
            init_fn=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        # Down-projection
        self.proj_down = Linear(
            in_features=config._inner_embedding_dim,
            out_features=config.embedding_dim,
            kernel_init=nnx.with_partitioning(
                wang_initializer(
                    dim=config.embedding_dim,
                    num_blocks=config._num_blocks,
                ),
                sharding=("tp", None),
                mesh=mesh,
            ),
        )

        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array):
        # Up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = jnp.split(
            x_inner,
            indices_or_sections=2,
            # indices_or_sections=self.inner_embedding_dim,
            axis=-1,
        )

        # mLSTM branch
        x_mlstm_conv = jax.nn.swish(self.conv1d(x_mlstm))  # SiLU is same as swish

        q = self.q_proj(x_mlstm_conv)
        k = self.k_proj(x_mlstm_conv)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state, learnable_skip, x_mlstm_conv = self.promote_dtype(
            (h_tilde_state, self.learnable_skip.value, x_mlstm_conv),
            dtype=self.dtype,
        )

        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv)

        # Output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # Down-projection with dropout
        y = self.dropout(self.proj_down(h_state))
        return y
