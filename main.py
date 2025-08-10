from functools import partial

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf

from moxe.config import MoxEConfig
from moxe.modules.model import MoxEForCausalLM
from moxe.utils.modules import count_parameters


@partial(nnx.jit, static_argnums=(0, 1))
def create_sharded_model(mesh: Mesh, config: MoxEConfig):
    rngs = nnx.Rngs(jax.random.key(123))
    model = MoxEForCausalLM(config, mesh=mesh, rngs=rngs, dtype=jnp.float32)

    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    config = MoxEConfig.from_dict(OmegaConf.to_container(cfg["model"], resolve=True))
    USE_JIT = True
    model = None

    dummy_input = jax.random.randint(
        jax.random.key(123),
        shape=(2, 5),
        minval=1,
        maxval=config.xlstm.vocab_size,
    )

    print("Creating device mesh")
    devices = mesh_utils.create_device_mesh((1, 1))
    mesh = Mesh(devices, axis_names=("dp", "tp"))

    print("Creating sharded model")
    with mesh:
        model = create_sharded_model(mesh, config)

    print(count_parameters(model))

    jitted_model = nnx.jit(model)

    output = None
    if USE_JIT:
        output = jitted_model(
            dummy_input,
            compute_d_loss=True,
            compute_group_loss=True,
            return_layers_outputs=True,
        )
    else:
        output = model(
            dummy_input,
            compute_d_loss=True,
            compute_group_loss=True,
            return_layers_outputs=True,
        )

    print(output.logits.shape)
    print(len(output.layers_outputs.z_loss.shape))


if __name__ == "__main__":
    main()
