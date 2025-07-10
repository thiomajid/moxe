"""
Simple test for the Gemma2 weight converter without full moxe imports.
"""

import sys
import typing as tp
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma2Config, Gemma2ForCausalLM, BertModel

from moxe.modules.hf.gemma2 import Gemma2ForCausalLM as FlaxGemma2ForCausalLM

# Import directly from the module
from moxe.modules.hf.gemma2_converter import Gemma2WeightConverter
from moxe.utils.array import create_mesh

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@partial(
    nnx.jit,
    static_argnames=("config_fn", "seed", "mesh", "dtype"),
)
def _create_sharded_model(
    config_fn: tp.Callable[[], Gemma2Config],
    seed: int,
    mesh: Mesh,
    dtype=jnp.float32,
):
    rngs = nnx.Rngs(seed)
    config = config_fn()
    model = FlaxGemma2ForCausalLM(config, mesh=mesh, rngs=rngs, dtype=dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


def test_weight_mapping():
    """Test that the weight mapping is correctly created."""
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        sliding_window=16,
        max_position_embeddings=100,
    )

    converter = Gemma2WeightConverter(config)
    mapping = converter._weight_mapping

    # Check some key mappings
    assert "model.embed_tokens.weight" in mapping
    assert "model.norm.weight" in mapping
    assert "lm_head.weight" in mapping
    assert "model.layers.0.self_attn.q_proj.weight" in mapping
    assert "model.layers.0.mlp.gate_proj.weight" in mapping

    torch_gemma = Gemma2ForCausalLM(config)

    mesh = create_mesh(mesh_shape=(1, 1), axis_names=("dp", "tp"))
    gemma = None
    with mesh:
        gemma = _create_sharded_model(
            config_fn=lambda: torch_gemma.config,
            seed=56,
            mesh=mesh,
            dtype=jnp.float32,
        )

    converted_gemma = converter.convert_weights(torch_gemma, gemma)

    print("✓ Weight mapping test passed")


def test_tensor_conversion():
    """Test PyTorch to JAX tensor conversion."""
    config = Gemma2Config(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        sliding_window=16,
        max_position_embeddings=100,
    )

    converter = Gemma2WeightConverter(config)

    # Test basic tensor conversion
    torch_tensor = torch.randn(10, 20)
    jax_array = converter._torch_to_jax(torch_tensor)

    assert isinstance(jax_array, jax.Array)
    assert jax_array.shape == (10, 20)

    # Test linear weight transposition
    linear_weight = torch.randn(20, 10)  # PyTorch format: [out, in]
    transposed = converter._transpose_linear_weight(linear_weight)

    assert transposed.shape == (10, 20)  # Flax format: [in, out]

    print("✓ Tensor conversion test passed")


def test_nested_attribute_access():
    """Test nested attribute getting and setting."""
    config = Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    converter = Gemma2WeightConverter(config)

    # Create a simple nested object for testing
    class TestObj:
        def __init__(self):
            self.level1 = type("obj", (object,), {})()
            self.level1.level2 = type("obj", (object,), {})()
            self.level1.level2.value = None

    obj = TestObj()

    # Test setting nested attribute
    test_value = jnp.array([1, 2, 3])
    converter._set_nested_attr(obj, "level1.level2.value", test_value)

    # Test getting nested attribute
    retrieved_value = converter._get_nested_attr(obj, "level1.level2.value")

    assert jnp.array_equal(retrieved_value, test_value)

    print("✓ Nested attribute access test passed")


def run_basic_tests():
    """Run basic tests that don't require the full model."""
    print("Running basic Gemma2 weight converter tests...\n")

    try:
        test_weight_mapping()
        test_tensor_conversion()
        test_nested_attribute_access()

        print("\n✅ All basic tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_basic_tests()
