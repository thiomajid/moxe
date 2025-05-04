# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano


import jax
import jax.numpy as jnp


@jax.jit
def slstm_forward_pointwise(
    Wx: jnp.ndarray,  # dim [B, 4*H]
    Ry: jnp.ndarray,  # dim [B, 4*H]
    b: jnp.ndarray,  # dim [1, 4*H]
    states: jnp.ndarray,  # dim [4, B, H]
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
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
    states_reshaped = states.reshape(4, states.shape[1], -1)
    y = states_reshaped[0]  # hidden state
    c = states_reshaped[1]  # cell state
    n = states_reshaped[2]  # normalization state
    m = states_reshaped[3]  # memory state

    # Split the raw activations into the 4 gates
    raw_reshaped = raw.reshape(raw.shape[0], 4, -1)
    iraw = raw_reshaped[:, 0]  # input gate
    fraw = raw_reshaped[:, 1]  # forget gate
    zraw = raw_reshaped[:, 2]  # cell update
    oraw = raw_reshaped[:, 3]  # output gate

    # Compute logfplusm
    logfplusm = m + jax.nn.log_sigmoid(fraw)

    # Replace conditional with a safe jax-compatible approach
    # Instead of using if-else based on tensor value, use jnp.where
    is_n_zero = jnp.all(n == 0.0)
    mnew = jax.lax.cond(is_n_zero, lambda: iraw, lambda: jnp.maximum(iraw, logfplusm))

    # Compute gate activations
    ogate = jax.nn.sigmoid(oraw)
    igate = jnp.exp(iraw - mnew)
    fgate = jnp.exp(logfplusm - mnew)
    cnew = fgate * c + igate * jnp.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    # Stack the new states and gate activations
    new_states = jnp.stack((ynew, cnew, nnew, mnew), axis=0)
    gate_activations = jnp.stack((igate, fgate, zraw, ogate), axis=0)

    return new_states, gate_activations
