import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma2Config

from moxe.modules.hf.gemma2 import Gemma2ForCausalLM
from moxe.utils.array import create_mesh
from moxe.utils.modules import count_parameters


@partial(
    nnx.jit,
    static_argnames=("config_fn", "seed", "mesh", "dtype"),
)
def _create_sharded_model(
    config_fn: tp.Callable[[], Gemma2Config],
    seed: int,
    mesh: Mesh,
    dtype=jnp.float32,
):
    rngs = nnx.Rngs(seed)
    config = config_fn()
    model = Gemma2ForCausalLM(config, mesh=mesh, rngs=rngs, dtype=dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


def main():
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        sliding_window=16,
        max_position_embeddings=100,
    )
    print(
        f"- Head dim: {config.head_dim}\n-Intermediate size: {config.intermediate_size}"
    )

    mesh = create_mesh(mesh_shape=(1, 1), axis_names=("dp", "tp"))

    model = None
    with mesh:
        model = _create_sharded_model(
            config_fn=lambda: config,
            seed=56,
            mesh=mesh,
            dtype=jnp.float32,
        )

    print(count_parameters(model))

    key = jax.random.key(45)
    key, subkey = jax.random.split(key)
    input_ids = jax.random.randint(
        subkey,
        shape=(2, 10),
        minval=1,
        maxval=config.vocab_size - 1,
    )

    attention_mask = jnp.ones_like(input_ids)

    logits = model(input_ids, attention_mask)
    print(logits.shape)


if __name__ == "__main__":
    main()
