import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from moxe.config import MoxEConfig
from moxe.modules.model import MoxEForCausalLM
from moxe.utils.modules import count_parameters


@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    config = MoxEConfig.from_dict(OmegaConf.to_container(cfg["model"], resolve=True))
    rngs = nnx.Rngs(123)

    USE_JIT = True

    model = MoxEForCausalLM(config, rngs=rngs, dtype=jnp.float32)
    print(count_parameters(model))

    jitted_model = nnx.jit(
        model,
        static_argnames=(
            "return_layers_outputs",
            "compute_d_loss",
            "compute_group_loss",
        ),
    )

    dummy_input = jax.random.randint(
        jax.random.key(123),
        shape=(2, 5),
        minval=1,
        maxval=config.xlstm.vocab_size,
    )

    output = None
    if USE_JIT:
        output = jitted_model(
            dummy_input,
            return_layers_outputs=True,
            compute_d_loss=True,
            compute_group_loss=True,
        )
    else:
        output = model(
            dummy_input,
            return_layers_outputs=True,
            compute_d_loss=True,
            compute_group_loss=True,
        )

    print(output.logits.shape)

    if output.layers_outputs is not None:
        print(f"Has {len(output.layers_outputs)} layers")


if __name__ == "__main__":
    main()
    # tree = (1, 2)
    # graph_def, state = nnx.split(tree)

    # print(nnx.merge(graph_def, state))
