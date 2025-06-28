# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_initializer, wang_initializer
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass
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
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        # Up-projection
        self.proj_up = nnx.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._inner_embedding_dim,
            use_bias=self.config.bias,
            kernel_init=lambda key, shape, param_dtype: small_init_initializer(
                dim=self.config.embedding_dim
            )(key, shape, param_dtype),
            bias_init=jax.nn.initializers.zeros,
            param_dtype=dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # QKV projections
        num_proj_heads = round(
            self.config._inner_embedding_dim // self.config.qkv_proj_blocksize
        )

        self.q_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            ),
            rngs=rngs,
            dtype=self.dtype,
        )

        self.k_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            ),
            rngs=rngs,
            dtype=self.dtype,
        )

        self.v_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            ),
            rngs=rngs,
            dtype=self.dtype,
        )

        # Convolutional layer
        self.conv1d = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
            ),
            rngs=rngs,
            dtype=self.dtype,
        )

        # mLSTM cell
        self.mlstm_cell = mLSTMCell(
            config=mLSTMCellConfig(
                context_length=self.config.context_length,
                embedding_dim=self.config._inner_embedding_dim,
                num_heads=self.config.num_heads,
            ),
            rngs=rngs,
            dtype=self.dtype,
        )

        self.ogate_act_fn = jax.nn.swish

        # Learnable skip connection parameter
        self.learnable_skip = nnx.Param(
            jnp.ones(self.config._inner_embedding_dim, dtype=self.dtype)
        )

        # Down-projection
        self.proj_down = nnx.Linear(
            in_features=self.config._inner_embedding_dim,
            out_features=self.config.embedding_dim,
            use_bias=self.config.bias,
            kernel_init=lambda key, shape, p_dtype: wang_initializer(
                dim=self.config.embedding_dim, num_blocks=self.config._num_blocks
            )(key, shape, dtype=p_dtype),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=self.dtype,
            dtype=self.dtype,
        )

        self.dropout = nnx.Dropout(rate=self.config.dropout, rngs=rngs)

    # @nnx.jit
    def __call__(self, x: jax.Array):
        """Forward pass for processing a full sequence.

        Args:
            x: Input tensor of shape (B, S, H)

        Returns:
            Output tensor of shape (B, S, H)
        """
        # B, S, _ = x.shape

        # Up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = jnp.split(
            x_inner, indices_or_sections=[self.config._inner_embedding_dim], axis=-1
        )

        # mLSTM branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = jax.nn.swish(x_mlstm_conv)  # SiLU is same as swish

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # Output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # Down-projection with dropout
        y = self.dropout(self.proj_down(h_state))
        return y

    def reset_parameters(self, rngs: nnx.Rngs):
        """Reset parameters of the layer."""
        # init inproj
        small_init_fn = small_init_initializer(dim=self.config.embedding_dim)
        self.proj_up.kernel = nnx.Param(
            small_init_fn(rngs.params(), self.proj_up.kernel.shape, self.dtype)
        )

        if self.proj_up.bias is not None:
            self.proj_up.bias = nnx.Param(
                nnx.initializers.zeros(
                    rngs.params(), self.proj_up.bias.shape, self.dtype
                )
            )

        wang_init_fn = wang_initializer(
            dim=self.config.embedding_dim, num_blocks=self.config._num_blocks
        )
        self.proj_down.kernel = nnx.Param(
            wang_init_fn(rngs.params(), self.proj_down.kernel.shape, self.dtype)
        )

        if self.proj_up.bias is not None:
            self.proj_down.bias = nnx.Param(
                nnx.initializers.zeros(
                    rngs.params(), self.proj_down.bias.shape, self.dtype
                )
            )

        self.learnable_skip = nnx.Param(
            nnx.initializers.ones(rngs.params(), self.learnable_skip.shape, self.dtype)
        )

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            qkv_proj.kernel = nnx.Param(
                small_init_fn(rngs.params(), qkv_proj.kernel.shape, self.dtype)
            )

            if qkv_proj.bias is not None:
                qkv_proj.bias = nnx.Param(
                    nnx.initializers.zeros(
                        rngs.params(), qkv_proj.bias.shape, self.dtype
                    )
                )

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters(rngs)
