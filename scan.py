import jax.numpy as jnp
from jax import lax


def pp(carry, idx):
    return carry + 1, {"carry": carry, "index": idx}


if __name__ == "__main__":
    xs = jnp.arange(5)
    init_carry = 0

    final, accumulation = lax.scan(f=pp, init=init_carry, xs=xs)

    print(final)
