# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import UpProjConfigMixin
from .init import small_init_initializer, wang_initializer

# Define activation functions using JAX
_act_fn_registry = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "relu^2": lambda x: jnp.square(jax.nn.relu(x)),
    "sigmoid": jax.nn.sigmoid,
    "swish": jax.nn.swish,
    "selu": jax.nn.selu,
}


def get_act_fn(act_fn_name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get activation function by name."""
    if act_fn_name in _act_fn_registry:
        return _act_fn_registry[act_fn_name]
    else:
        raise ValueError(
            f'Unknown activation function name "{act_fn_name}". '
            f"Available activation functions are: {list(_act_fn_registry.keys())}"
        )


@dataclass
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    ff_type: Literal["ffn_gated"] = "ffn_gated"

    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        assert self.act_fn in _act_fn_registry, (
            f"Unknown activation function {self.act_fn}"
        )


class GatedFeedForward(nnx.Module):
    config_class = FeedForwardConfig

    def __init__(
        self,
        config: FeedForwardConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config

        # Initialize linear layers
        self.proj_up = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=2 * config._proj_up_dim,
            use_bias=config.bias,
            kernel_init=small_init_initializer(dim=config.embedding_dim),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        self.proj_down = nnx.Linear(
            in_features=config._proj_up_dim,
            out_features=config.embedding_dim,
            use_bias=config.bias,
            kernel_init=wang_initializer(
                dim=config.embedding_dim, num_blocks=config._num_blocks
            ),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        self.act_fn = get_act_fn(config.act_fn)
        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        # Project up and split into gate and activation path
        up_proj_output = self.proj_up(x)
        gate_preact, up_proj = jnp.split(
            up_proj_output,
            indices_or_sections=[self.config._proj_up_dim],
            axis=-1,
        )

        activated = self.act_fn(gate_preact) * up_proj
        output = self.dropout(self.proj_down(activated))

        return output

    def reset_parameters(self, rngs: nnx.Rngs):
        """Reset parameters of the linear layers."""
        # Initialize weights using small init for proj_up
        small_init = small_init_initializer(dim=self.config.embedding_dim)

        self.proj_up.kernel = nnx.Param(
            small_init(
                key=rngs.params(),
                shape=self.proj_up.kernel.shape,
                dtype=self.proj_up.kernel.dtype,
            )
        )

        # Initialize weights using Wang init for proj_down
        wang_init = wang_initializer(
            dim=self.config.embedding_dim,
            num_blocks=self.config._num_blocks,
        )

        self.proj_down.kernel = nnx.Param(
            wang_init(
                key=rngs.params(),
                shape=self.proj_down.kernel.shape,
                dtype=self.proj_down.kernel.dtype,
            )
        )

        # Initialize biases to zero
        if self.proj_up.bias is not None:
            self.proj_up.bias = nnx.Param(
                jax.nn.initializers.zeros(
                    key=rngs.params(),
                    shape=self.proj_up.bias.shape,
                    dtype=self.proj_up.bias.dtype,
                )
            )

        if self.proj_down.bias is not None:
            self.proj_down.bias = nnx.Param(
                jax.nn.initializers.zeros(
                    key=rngs.params(),
                    shape=self.proj_down.bias.shape,
                    dtype=self.proj_down.bias.dtype,
                )
            )


def create_feedforward(config: FeedForwardConfig, rngs: nnx.Rngs, dtype=jnp.float32):
    """Factory function to create feedforward modules based on config."""
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config, rngs=rngs, dtype=dtype)
    else:
        raise ValueError(f"Unknown feedforward type {config.ff_type}")
