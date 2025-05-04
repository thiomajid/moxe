# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass, field
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nnx.Module):
    config_class = CausalConv1dConfig
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(
        self,
        config: CausalConv1dConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.dtype = dtype
        self.groups = self.config.feature_dim
        if self.config.channel_mixing:
            self.groups = 1

        if self.config.kernel_size == 0:
            # No convolution needed
            self.conv = None
        else:
            # padding of this size assures temporal causality.
            self.pad = self.config.kernel_size - 1

            self.conv = nnx.Conv(
                in_features=self.config.feature_dim,
                out_features=self.config.feature_dim,
                kernel_size=(self.config.kernel_size,),
                padding=[(self.pad, 0)],
                feature_group_count=self.groups,
                kernel_init=jax.nn.initializers.lecun_normal(),
                use_bias=self.config.causal_conv_bias,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
                dtype=self.dtype,
            )

    def __call__(self, x: jnp.ndarray):
        if self.config.kernel_size == 0:
            return x

        # With nnx.Conv the feature dimension is the last one so no need to transpose
        return self.conv(x)  # (B, T+pad, F) tensor

    def reset_parameters(self, rngs: nnx.Rngs):
        """Reset the parameters of the convolutional layer."""
        if self.config.kernel_size == 0:
            return

        # Reset kernel
        kernel_init = jax.nn.initializers.lecun_normal()
        self.conv.kernel = nnx.Param(
            kernel_init(
                rngs.params(),
                self.conv.kernel.value.shape,
                self.conv.kernel.value.dtype,
            )
        )

        # Reset bias if it exists
        if self.config.causal_conv_bias:
            bias_init = jax.nn.initializers.zeros
            self.conv.bias = nnx.Param(
                bias_init(
                    rngs.params(),
                    self.conv.bias.value.shape,
                    self.conv.bias.value.dtype,
                )
            )

    def _create_weight_decay_optim_groups(
        self,
    ) -> Tuple[Tuple[nnx.Param, ...], Tuple[nnx.Param, ...]]:
        if self.config.kernel_size == 0:
            return (), ()
        else:
            weight_decay = (self.conv.kernel,)
            no_weight_decay = ()
            if self.config.causal_conv_bias and self.bias is not None:
                no_weight_decay = (self.conv.bias,)
            return weight_decay, no_weight_decay
