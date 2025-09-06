# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel
# Ported to JAX/Flax by Abdoul Majid O. Thiombiano


from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


@partial(jax.jit, static_argnames=("mesh",))
def slstm_forward_pointwise(
    Wx: jax.Array,  # dim [B, 4*H]
    Ry: jax.Array,  # dim [B, 4*H]
    b: jax.Array,  # dim [1, 4*H]
    states: jax.Array,  # dim [4, B, H]
    mesh: Mesh,
) -> tuple[
    jax.Array,
    jax.Array,
]:
    """
    Implements the sLSTM forward pass operation in JAX.

    Args:
        Wx: Input projections with shape [B, 4*H]
        Ry: Recurrent projections with shape [B, 4*H]
        b: Bias terms with shape [1, 4*H]
        states: Previous states with shape [4, B, H]

    Returns:
        Tuple containing:
            - New states [4, B, H] (hidden, cell, n, m)
            - Gate activations [4, B, H] (input, forget, cell, output)
    """

    # Combine input projection, recurrent projection, and bias
    raw = Wx + Ry + b

    # Extract states from the states tensor
    states = states.reshape(4, states.shape[1], -1)
    # y = states[0]  # hidden state
    c = states[1]  # cell state
    n = states[2]  # normalization state
    m = states[3]  # memory state

    # Split the raw activations into the 4 gates
    raw = raw.reshape(raw.shape[0], 4, -1)
    iraw = raw[:, 0]
    fraw = raw[:, 1]
    zraw = raw[:, 2]
    oraw = raw[:, 3]
    # raw = jax.lax.with_sharding_constraint(
    #     raw,
    #     NamedSharding(mesh, P(None, None, "tp")),
    # )

    # with mesh:
    #     iraw = jax.lax.with_sharding_constraint(raw[:, 0], P("dp", "tp"))  # input gate

    #     fraw = jax.lax.with_sharding_constraint(raw[:, 1], P("dp", "tp"))  # forget gate
    #     zraw = jax.lax.with_sharding_constraint(raw[:, 2], P("dp", "tp"))  # cell update
    #     oraw = jax.lax.with_sharding_constraint(raw[:, 3], P("dp", "tp"))  # output gate

    logfplusm = m + jax.nn.log_sigmoid(fraw)

    mnew = jax.lax.cond(
        jnp.all(n == 0.0),
        lambda: iraw,
        lambda: jnp.maximum(iraw, logfplusm),
    )

    ogate = jax.nn.sigmoid(oraw)
    igate = jnp.minimum(jnp.exp(iraw - mnew), jnp.ones_like(iraw))
    fgate = jnp.minimum(jnp.exp(logfplusm - mnew), jnp.ones_like(iraw))
    cnew = fgate * c + igate * jnp.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    # Stack the new states and gate activations
    new_states = jnp.stack((ynew, cnew, nnew, mnew), axis=0)
    gate_activations = jnp.stack((igate, fgate, zraw, ogate), axis=0)

    return new_states, gate_activations
