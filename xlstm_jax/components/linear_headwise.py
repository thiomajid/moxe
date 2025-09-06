# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbininan Pöppel
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.nnx.nn import dtypes


@dataclass(unsafe_hash=True, order=True)
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

    def __init__(
        self,
        config: LinearHeadwiseExpandConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        kernel_init=initializers.lecun_normal(),
        bias_init=initializers.zeros_init(),
    ):
        in_features = config.in_features
        num_heads = config.num_heads
        in_features_per_head = in_features // num_heads
        out_features_per_head = config._out_features // num_heads

        self.num_heads = num_heads
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.promote_dtype = dtypes.promote_dtype

        kernel_shape = (num_heads, in_features_per_head, out_features_per_head)
        self.kernel = nnx.Param(kernel_init(rngs.params(), kernel_shape, param_dtype))

        # Create bias parameter
        if config.bias:
            self.bias = nnx.Param(
                bias_init(rngs.params(), (config._out_features), param_dtype)
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        shape = x.shape
        x = x.reshape(*shape[:-1], self.num_heads, -1)

        kernel = self.kernel.value
        bias = self.bias.value if self.bias else None
        x, kernel, bias = self.promote_dtype((x, kernel, bias), dtype=self.dtype)

        # with flax, the kernel is not transposed, out_features is the last dimension
        x = jnp.einsum("...hd,hdo->...ho", x, kernel)
        x = x.reshape(*shape[:-1], -1)

        if bias is not None:
            x = x + bias

        return x
