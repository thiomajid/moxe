import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from moxe.config import MoxEConfig
from moxe.modules.model import MoxEForCausalLM


@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    config = MoxEConfig.from_dict(OmegaConf.to_container(cfg["model"], resolve=True))

    rngs = nnx.Rngs(123)
    model = MoxEForCausalLM(config, rngs=rngs, dtype=jnp.float32)

    dummy_input = jax.random.randint(
        jax.random.key(123),
        shape=(2, 5),
        minval=1,
        maxval=config.xlstm.vocab_size,
    )

    output = model(
        dummy_input,
        return_layers_outputs=False,
        compute_d_loss=True,
        compute_group_loss=True,
    )

    print(output.logits.shape)


if __name__ == "__main__":
    main()
