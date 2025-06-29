import os
from functools import partial

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf
from xlstm import xLSTMLMModel

from moxe.config import MoxEConfig
from moxe.utils.modules import count_parameters
from xlstm_jax import xLSTMLMModel as FlaxxLSTM


@partial(nnx.jit, static_argnums=(0, 1))
def create_sharded_model(mesh: Mesh, config: MoxEConfig):
    rngs = nnx.Rngs(jax.random.key(123))
    model = FlaxxLSTM(config.xlstm, mesh=mesh, rngs=rngs, dtype=jnp.float32)

    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    config = MoxEConfig.from_dict(OmegaConf.to_container(cfg["model"], resolve=True))
    USE_JIT = False
    model = None

    dummy_input = jax.random.randint(
        jax.random.key(123),
        shape=(2, 5),
        minval=1,
        maxval=config.xlstm.vocab_size,
    )

    t_input = torch.from_numpy(np.array(jax.device_get(dummy_input)))
    t_model = xLSTMLMModel(config=config.xlstm)
    t_output = t_model(t_input)
    print(f"Torch output {t_output.shape}")

    print("Creating device mesh")
    devices = mesh_utils.create_device_mesh((2, 4))
    mesh = Mesh(devices, axis_names=("dp", "tp"))

    print("Creating sharded model")
    with mesh:
        model = create_sharded_model(mesh, config)

    print(count_parameters(model))

    jitted_model = nnx.jit(
        model,
        # static_argnames=(
        #     "return_layers_outputs",
        #     "compute_d_loss",
        #     "compute_group_loss",
        # ),
    )

    output = None
    if USE_JIT:
        output = jitted_model(
            dummy_input,
            # return_layers_outputs=True,
            # compute_d_loss=True,
            # compute_group_loss=True,
        )
    else:
        output = model(
            dummy_input,
            # return_layers_outputs=True,
            # compute_d_loss=True,
            # compute_group_loss=True,
        )

    print(output.shape)

    # if output.layers_outputs is not None:
    #     print(f"Has {len(output.layers_outputs)} layers")


if __name__ == "__main__":
    main()
    # tree = (1, 2)
    # graph_def, state = nnx.split(tree)

    # print(nnx.merge(graph_def, state))
