# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano


import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh


def LayerNorm(
    num_features: int,
    mesh: Mesh,
    rngs: nnx.Rngs,
    use_scale: bool = True,
    use_bias: bool = False,
    epsilon: float = 1e-5,
    dtype=jnp.float32,
):
    return nnx.LayerNorm(
        num_features=num_features,
        use_scale=use_scale,
        use_bias=use_bias,
        rngs=rngs,
        dtype=dtype,
        param_dtype=dtype,
        epsilon=epsilon,
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


def MultiHeadLayerNorm(
    num_features: int,
    mesh: Mesh,
    rngs: nnx.Rngs,
    axis: int = 1,
    use_scale: bool = True,
    use_bias: bool = False,
    epsilon: float = 1e-5,
    dtype=jnp.float32,
):
    return nnx.vmap(
        LayerNorm(
            num_features=num_features,
            mesh=mesh,
            rngs=rngs,
            use_scale=use_scale,
            use_bias=use_bias,
            epsilon=epsilon,
            dtype=dtype,
        ),
        in_axes=axis,
        out_axes=axis,
    )
