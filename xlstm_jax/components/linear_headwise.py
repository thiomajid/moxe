# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from xlstm_jax.components.init import small_init_initializer


@dataclass
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    # this is the number of heads that the in_features are split into
    # if num_heads=1, this is a normal linear layer
    # if num_heads>1, the in_features are split into num_heads and each head is projected separately
    # if num_heads=in_features, each feature is projected separately
    num_heads: int = -1
    expand_factor_up: float = 1

    # this is internally computed
    # but can be overwritten if you want to use a different output dimension
    # if > 0 the expand factor is ignored
    _out_features: int = -1

    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, (
            "in_features must be a multiple of num_heads"
        )

        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nnx.Module):
    """This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    config_class = LinearHeadwiseExpandConfig

    def __init__(
        self,
        config: LinearHeadwiseExpandConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        in_features = config.in_features
        num_heads = config.num_heads
        in_features_per_head = in_features // num_heads
        out_features_per_head = config._out_features // num_heads

        # Create weight parameter
        # stddev = math.sqrt(2 / 5 / in_features_per_head)
        if config.trainable_weight:
            self.kernel = nnx.Param(
                jnp.empty(
                    (num_heads, out_features_per_head, in_features_per_head),
                    dtype=dtype,
                ),
                # init_fn=jax.nn.initializers.normal(stddev=stddev),
                init_fn=nnx.with_partitioning(
                    small_init_initializer(in_features_per_head),
                    sharding=(None, None, "model"),
                    mesh=mesh,
                ),
            )
        else:
            # For non-trainable weights, use nnx.State instead of nnx.Param
            self.kernel = nnx.State(
                jnp.empty(
                    (num_heads, out_features_per_head, in_features_per_head),
                    dtype=dtype,
                ),
                init_fn=nnx.with_partitioning(
                    small_init_initializer(in_features_per_head),
                    sharding=(None, None, "model"),
                    mesh=mesh,
                ),
            )

        # Create bias parameter
        if config.bias:
            if config.trainable_bias:
                self.bias = nnx.Param(
                    jnp.zeros(config._out_features, dtype=dtype),
                    init_fn=nnx.with_partitioning(
                        nnx.initializers.zeros_init(),
                        sharding=("model",),
                        mesh=mesh,
                    ),
                )
            else:
                self.bias = nnx.State(
                    jnp.zeros(config._out_features, dtype=dtype),
                    init_fn=nnx.with_partitioning(
                        nnx.initializers.zeros_init(),
                        sharding=("model",),
                        mesh=mesh,
                    ),
                )
        else:
            self.bias = None

        self.num_heads = num_heads

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for the headwise linear transformation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Get shape information
        shape = x.shape
        x = x.reshape(*shape[:-1], self.num_heads, -1)
        x = jnp.einsum("...hd,hod->...ho", x, self.kernel)
        x = x.reshape(*shape[:-2], -1)

        if self.bias is not None:
            x = x + self.bias

        # x = jax.lax.cond(
        #     self.bias is not None,
        #     lambda _x: _x + self.bias,
        #     lambda _x: _x,
        #     operand=x,
        # )

        return x
