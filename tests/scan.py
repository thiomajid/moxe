import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


def pp(carry, idx):
    return carry + 1, {"carry": carry, "index": idx}


@nnx.vmap
def f(x):
    jax.debug.print("{}", x)


if __name__ == "__main__":
    xs = jnp.arange(5)
    init_carry = 0

    final, accumulation = lax.scan(f=pp, init=init_carry, xs=xs)

    print(final)

    # f(jnp.arange(10))

    rngs = nnx.Rngs(123)

    dummy = jax.random.normal(rngs(), shape=(5, 1))
    layers = [nnx.Linear(1, 1, rngs=rngs) for _ in range(5)]

    # Split the layers into static and variable parts
    split_out = [nnx.split(layer) for layer in layers]
    graphdefs = [el[0] for el in split_out]
    states = [el[1] for el in split_out]

    var_layers = jax.tree.map(lambda *args: jnp.stack(args), *states)
    var_graphs = jax.tree.map(lambda *args: jnp.stack(args), *graphdefs)

    def _linear_scan(carry, inputs):
        graphdef, var_layer = inputs
        # graphdef = graphdef[0]
        jax.debug.print("here")
        layer = nnx.merge(graphdef, var_layer)
        new_state = layer(carry)
        return new_state, new_state

    final2, stack = lax.scan(f=_linear_scan, init=dummy, xs=(var_graphs, var_layers))
    print(final2)

    fin3 = dummy
    for layer in layers:
        fin3 = layer(fin3)

    diff = fin3 - final2
    print(jnp.allclose(fin3, final2), diff)
