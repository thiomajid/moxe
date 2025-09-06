# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano
import math
import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...components.ln import MultiHeadLayerNorm
from .cell import sLSTMCell, sLSTMCellConfig


@dataclass(unsafe_hash=True, order=True)
class sLSTMLayerConfig(sLSTMCellConfig):
    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the hidden size, is not yet supported by all versions in this directory
    conv1d_kernel_size: int = 4  # 0 means no convolution included
    group_norm_weight: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        self.hidden_size = self.embedding_dim
        sLSTMCellConfig.__post_init__(self)


class sLSTMLayer(nnx.Module):
    """sLSTM layer implementation in JAX/Flax."""

    conv1d: tp.Callable[[jax.Array], jax.Array] | CausalConv1d

    def __init__(
        self,
        config: sLSTMLayerConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.conv1d_kernel_size = config.conv1d_kernel_size

        # Initialize convolutional layer if needed
        if config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                param_dtype=param_dtype,
                config=CausalConv1dConfig(
                    feature_dim=config.embedding_dim,
                    kernel_size=config.conv1d_kernel_size,
                ),
            )
        else:
            self.conv1d = jax.nn.identity

        self.conv_act_fn = jax.nn.swish

        # Initialize gate projections using headwise linear layers
        gate_config = LinearHeadwiseExpandConfig(
            in_features=config.embedding_dim,
            num_heads=config.num_heads,
            bias=False,
        )

        Gate = partial(
            LinearHeadwiseExpand,
            config=gate_config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                initializer=nnx.initializers.normal(
                    math.sqrt(2 / 5 / gate_config.in_features)
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

        self.fgate = Gate()
        self.igate = Gate()
        self.zgate = Gate()
        self.ogate = Gate()

        # sLSTM cell and normalization
        self.slstm_cell = sLSTMCell(
            config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.group_norm = MultiHeadLayerNorm(
            num_features=config.embedding_dim,
            use_scale=config.group_norm_weight,
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

        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        slstm_state: tp.Optional[jax.Array] = None,
        return_last_state: bool = False,
    ) -> tp.Union[jax.Array, tuple[jax.Array, dict[str, tp.Any]]]:
        """Process a sequence through the sLSTM layer.

        Args:
            x: Input tensor of shape (B, S, D)
            conv_state: Initial convolution state
            slstm_state: Initial sLSTM cell state
            return_last_state: Whether to return final states

        Returns:
            Output tensor or tuple of output tensor and final states
        """

        x_conv = self.conv_act_fn(self.conv1d(x))

        f = self.fgate(x_conv)
        i = self.igate(x_conv)
        z = self.zgate(x)
        o = self.ogate(x)

        gates_combined = jnp.concatenate([i, f, z, o], axis=-1)
        y, slstm_state = self.slstm_cell(gates_combined, state=slstm_state)
        y = self.dropout(y)

        out = self.group_norm(y)

        if return_last_state:
            return out, {"slstm_state": slstm_state}
        else:
            return out
