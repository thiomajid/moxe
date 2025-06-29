import jax
import jax.numpy as jnp
from flax import nnx

from moxe.config import MoxEConfig


class FeedForwardExpert(nnx.Module):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.hidden_dim = config.xlstm.embedding_dim
        self.ffn_dim = 2 * self.hidden_dim

        self.w1 = nnx.Linear(
            self.hidden_dim,
            self.ffn_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.w2 = nnx.Linear(
            self.ffn_dim,
            self.hidden_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.w3 = nnx.Linear(
            self.hidden_dim,
            self.ffn_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.activation = jax.nn.silu
        self.mesh = mesh

    def __call__(self, x: jax.Array):
        h_t = self.activation(self.w1(x)) * self.w3(x)
        h_t = self.w2(h_t)
        return h_t
