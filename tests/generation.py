import functools
import random

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from moxe.config import MoxEConfig
from moxe.inference import GenerationCarry, generate_sequence_scan
from moxe.modules.model import MoxEForCausalLM
from moxe.utils.array import create_mesh


@functools.partial(
    nnx.jit,
    static_argnames=("config_fn", "seed", "mesh", "dtype"),
)
def _create_sharded_model(
    config_fn: callable,
    seed: int,
    mesh: Mesh,
    dtype=jnp.float32,
):
    rngs = nnx.Rngs(seed)
    config = config_fn()
    model = MoxEForCausalLM(config, mesh=mesh, rngs=rngs, dtype=dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


@hydra.main(
    config_path="../configs", config_name="train_moxe_config", version_base="1.1"
)
def main(cfg: DictConfig):
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 0.785
    GREEDY = False

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"

    config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config_dict["xlstm"]["vocab_size"] = tokenizer.vocab_size
    config_dict["xlstm"]["pad_token_id"] = tokenizer.pad_token_id
    config = MoxEConfig.from_dict(config_dict)

    mesh = create_mesh(mesh_shape=(1, 1), axis_names=("dp", "tp"))

    model = None
    with mesh:
        model = _create_sharded_model(
            config_fn=lambda: config,
            seed=56,
            mesh=mesh,
            dtype=jnp.float32,
        )

    # Generate some text
    GENERATION_SAMPLES = ["Once upon a time", "There was a girl", "Next to the tree"]
    choosen_prompt = random.choice(GENERATION_SAMPLES)
    input_ids = tokenizer(choosen_prompt, return_tensors="jax", padding=True)[
        "input_ids"
    ]

    batch_size = input_ids.shape[0]
    initial_len = input_ids.shape[1]
    total_length = initial_len + MAX_NEW_TOKENS
    full_x_init = jnp.zeros((batch_size, total_length), dtype=jnp.int32)
    full_x_init = full_x_init.at[:, :initial_len].set(input_ids)
    key = jax.random.key(123)
    initial_carry: GenerationCarry = (
        full_x_init,
        initial_len,
        key,
    )

    sequences = generate_sequence_scan(
        model,
        initial_carry,
        max_new_tokens=MAX_NEW_TOKENS,
        vocab_size=config.xlstm.vocab_size,
        temperature=TEMPERATURE,
        greedy=GREEDY,
    )

    sequences = tokenizer.batch_decode(sequences)
    print(sequences)

    _ = generate_sequence_scan(
        model,
        initial_carry,
        max_new_tokens=MAX_NEW_TOKENS,
        vocab_size=config.xlstm.vocab_size,
        temperature=TEMPERATURE,
        greedy=GREEDY,
    )


if __name__ == "__main__":
    main()
