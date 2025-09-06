import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# the model's State (its parameters and variables) must now be part of
# the carry to be threaded through the scan loop correctly.
GenerationCarry = Tuple[
    jax.Array,  # The full sequence array being built
    int,  # The current index to write the next token
    jax.Array,  # The current PRNG key
    nnx.State,  # The Pytree of the model's parameters
]


@functools.partial(
    nnx.jit,
    static_argnames=("max_new_tokens", "vocab_size", "temperature", "greedy"),
)
def generate_sequence_scan(
    model: nnx.Module,
    initial_carry_val: tuple[jax.Array, int, jax.Array],  # initial carry from caller
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 1.0,
    greedy: bool = False,
):
    """
    Generates a sequence using jax.lax.scan.

    This version is refactored to be robust with JAX's JIT compiler by using
    the standard "split-merge" pattern for NNX modules inside control flow.
    """
    # split the model into its GraphDef and State before entering the scan loop.
    # lax.scan also trace arrays, therefore we need to split the model otherwise the
    # jit tracing will conflict with the scan one, and scan can't operate on abstract arrays
    # because it needs to create its own traced arrays
    graphdef, state = nnx.split(model)

    # Augment the initial carry with the model's state
    initial_full_carry: GenerationCarry = (
        initial_carry_val[0],
        initial_carry_val[1],
        initial_carry_val[2],
        state,
    )

    # define the body function for lax.scan inside the jiited function.
    # this allows it to form a closure over the static `graphdef`.
    def _generation_step_body(carry: GenerationCarry, _):
        current_full_x, current_index, current_key, model_state = carry
        step_key, next_key = jax.random.split(current_key)

        # reconstruct the model for this step by merging the
        # static graphdef with the current state from the carry.
        model_for_step = nnx.merge(graphdef, model_state)
        out = model_for_step(current_full_x)
        batch_size = current_full_x.shape[0]

        # --- Logit Selection ---
        logits_slice = jax.lax.dynamic_slice(
            out,
            start_indices=(0, current_index - 1, 0),
            slice_sizes=(batch_size, 1, vocab_size),
        )

        last_token_logits = jnp.squeeze(logits_slice, axis=1)

        # --- Sampling Logic ---
        def _greedy_sample(logits, temp, key):
            return jnp.argmax(logits, axis=-1)

        def _temperature_sample(logits, temp, key):
            scaled_logits = logits / temp
            return jax.random.categorical(key, scaled_logits, axis=-1)

        next_token = jax.lax.cond(
            greedy,
            _greedy_sample,
            _temperature_sample,
            last_token_logits,
            temperature,
            step_key,
        )
        next_token = next_token.astype(jnp.int32)

        # --- State Update ---
        updated_full_x = jax.lax.dynamic_update_slice(
            current_full_x, next_token[:, None], (0, current_index)
        )

        # return the new carry, threading the (unchanged) model_state
        # through to the next iteration.
        return (updated_full_x, current_index + 1, next_key, model_state), next_token

    # --- Run the Scan ---
    final_carry, _ = jax.lax.scan(
        _generation_step_body,
        initial_full_carry,  # augmented carry
        None,
        length=max_new_tokens,
    )
    final_x = final_carry[0]
    return final_x


class GenerationMixin:
    vocab_size: int

    def __init__(self):
        pass

    def generate(
        self,
        initial_carry_val: Tuple[jax.Array, int, jax.Array],
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> jax.Array:
        return generate_sequence_scan(
            self,
            initial_carry_val,
            max_new_tokens,
            self.vocab_size,
            temperature,
            greedy,
        )
