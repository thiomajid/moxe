import os
from time import perf_counter

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from moxe.utils.modules import count_parameters
from xlstm_jax.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.blocks.slstm.block import sLSTMBlockConfig
from xlstm_jax.blocks.slstm.layer import sLSTMLayerConfig
from xlstm_jax.components.feedforward import FeedForwardConfig
from xlstm_jax.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig


def create_test_config() -> xLSTMLMModelConfig:
    # mLSTM layer configuration
    mlstm_layer_config = mLSTMLayerConfig(
        conv1d_kernel_size=4,
        qkv_proj_blocksize=8,  # Smaller blocksize for small embedding dim
        num_heads=2,
        proj_factor=2.0,
        embedding_dim=32,
        bias=False,
        dropout=0.0,
        context_length=16,
    )

    # mLSTM block configuration
    mlstm_block_config = mLSTMBlockConfig(
        mlstm=mlstm_layer_config,
    )

    # sLSTM layer configuration
    slstm_layer_config = sLSTMLayerConfig(
        embedding_dim=32,
        num_heads=2,
        conv1d_kernel_size=4,
        dropout=0.0,
        backend="cuda",
        bias_init="powerlaw_blockdependent",
    )

    # Feedforward configuration for sLSTM blocks
    feedforward_config = FeedForwardConfig(
        proj_factor=1.7,
        act_fn="swish",
        embedding_dim=32,
        dropout=0.0,
        bias=False,
    )

    # sLSTM block configuration
    slstm_block_config = sLSTMBlockConfig(
        slstm=slstm_layer_config,
        feedforward=feedforward_config,
    )

    # Main model configuration
    # Use alternating pattern: mLSTM at positions 0,2 and sLSTM at positions 1,3
    config = xLSTMLMModelConfig(
        vocab_size=1000,  # Small vocab for testing
        context_length=64,
        num_blocks=4,
        embedding_dim=32,
        add_post_blocks_norm=True,
        bias=False,
        dropout=0.0,
        slstm_at=[1, 3],  # sLSTM blocks at positions 1 and 3
        mlstm_block=mlstm_block_config,
        slstm_block=slstm_block_config,
        tie_weights=False,
        weight_decay_on_embedding=False,
        add_embedding_dropout=False,
        pad_token_id=0,
    )

    return config


def create_model(config: xLSTMLMModelConfig, mesh: Mesh) -> xLSTMLMModel:
    rngs = nnx.Rngs(jax.random.key(42))

    model = xLSTMLMModel(
        config=config,
        mesh=mesh,
        rngs=rngs,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    return model


def test_inference(
    model: xLSTMLMModel, config: xLSTMLMModelConfig, num_samples: int = 3
):
    print("Testing inference...")

    # Create test input sequences
    batch_size = 2
    seq_length = config.context_length

    # Generate random token sequences
    key = jax.random.key(123)
    input_ids = jax.random.randint(
        key,
        shape=(batch_size, seq_length),
        minval=1,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input tokens:\n{input_ids}")

    # Run inference
    start_time = perf_counter()
    logits = model(input_ids)
    end_time = perf_counter()

    print(f"\nInference completed in {end_time - start_time:.4f} seconds")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {config.vocab_size})")

    # Convert logits to probabilities and get top predictions
    probs = jax.nn.softmax(logits, axis=-1)
    top_tokens = jnp.argmax(probs, axis=-1)

    print(f"\nTop predicted tokens:\n{top_tokens}")

    # Show some probability values for the first sequence, last position
    last_probs = probs[0, -1, :]
    top_5_indices = jnp.argsort(last_probs)[-5:]
    print("\nTop 5 token probabilities for first sequence, last position:")
    for i, idx in enumerate(reversed(top_5_indices)):
        prob = last_probs[idx]
        print(f"  Token {idx}: {prob:.4f}")

    return logits


def generate_text(
    model: xLSTMLMModel,
    config: xLSTMLMModelConfig,
    prompt_tokens: jax.Array,
    num_new_tokens: int = 10,
):
    print(f"\nGenerating {num_new_tokens} new tokens...")

    current_tokens = prompt_tokens.copy()
    generated_tokens = []

    for step in range(num_new_tokens):
        # Get the last context_length tokens if sequence is too long
        if current_tokens.shape[-1] > config.context_length:
            input_tokens = current_tokens[:, -config.context_length :]
        else:
            input_tokens = current_tokens

        # Pad if sequence is too short
        if input_tokens.shape[-1] < config.context_length:
            pad_length = config.context_length - input_tokens.shape[-1]
            padding = jnp.zeros((input_tokens.shape[0], pad_length), dtype=jnp.int32)
            input_tokens = jnp.concatenate([padding, input_tokens], axis=-1)

        # Get model predictions
        logits = model(input_tokens)

        # Sample next token (using greedy decoding for simplicity)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        # Append to sequence
        current_tokens = jnp.concatenate([current_tokens, next_token], axis=-1)
        generated_tokens.append(next_token[0, 0])

        print(f"Step {step + 1}: Generated token {next_token[0, 0]}")

    print(f"Generated sequence: {generated_tokens}")
    return current_tokens


def main():
    """Main function to run the xLSTM inference test."""
    print("=== xLSTM Language Model Inference Test ===")
    print("Configuration:")
    print("- Embedding dim: 32")
    print("- Context length: 16")
    print("- Number of blocks: 4")
    print("- Block pattern: mLSTM at [0,2], sLSTM at [1,3]")
    print("- Heads per block: 2")
    print("- Vocab size: 1000")
    print()

    # Set up JAX for CPU (remove XLA_FLAGS for GPU)
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    # Create configuration
    config = create_test_config()
    print("✓ Configuration created")

    # Create mesh for model parallelism (using 1 device for simplicity)
    devices = mesh_utils.create_device_mesh((1, 1))
    mesh = Mesh(devices, axis_names=("dp", "tp"))

    print("✓ Device mesh created")

    # Create model
    with mesh:
        model = create_model(config, mesh)

    print("✓ Model created")

    # Count parameters
    print(f"✓ Model initialized with {count_parameters(model).millions} parameters")

    # Test basic inference
    print("\n" + "=" * 50)
    test_inference(model, config)

    # Test text generation
    print("\n" + "=" * 50)
    # Create a simple prompt
    prompt = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)  # Simple token sequence
    print(f"Starting with prompt tokens: {prompt[0].tolist()}")

    start = perf_counter()
    final_sequence = generate_text(model, config, prompt, num_new_tokens=30)
    total = perf_counter() - start

    print(f"Sequence generated in {total:.2f} seconds")
    print(f"Final sequence: {final_sequence[0].tolist()}")

    print("\n✓ Inference test completed successfully!")


if __name__ == "__main__":
    main()
