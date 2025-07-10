"""
Example usage of the Gemma2 weight converter.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from moxe.modules.hf.gemma2_converter import Gemma2WeightConverter


def main():
    """Example of how to convert HuggingFace Gemma2 weights to Flax."""

    # Setup JAX sharding
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("tp",))

    # Initialize RNG
    rngs = nnx.Rngs(0)

    # Model name - replace with actual Gemma2 model
    model_name = "google/gemma-2b"  # Example model name

    try:
        # Convert model from HuggingFace to Flax
        flax_model = Gemma2WeightConverter.from_pretrained(
            model_name_or_path=model_name,
            mesh=mesh,
            rngs=rngs,
            dtype=jnp.float32,
        )

        # Test the model with dummy input
        batch_size, seq_len = 2, 10
        dummy_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        dummy_attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

        # Run forward pass
        logits = flax_model(dummy_input_ids, dummy_attention_mask)
        print(f"Output shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")

        # Save the converted model
        save_path = "./converted_gemma2_flax"
        converter = Gemma2WeightConverter(flax_model.config)
        converter.save_converted_model(flax_model, save_path)

        print("Conversion successful!")

    except Exception as e:
        print(f"Error during conversion: {e}")
        print(
            "Note: Make sure you have access to the Gemma2 model and required dependencies."
        )


if __name__ == "__main__":
    main()
