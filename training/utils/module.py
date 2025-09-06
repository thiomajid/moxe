import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree as jtu
import orbax.checkpoint as ocp
from flax import nnx

_dtype_map = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


def str2dtype(dtype_str: str) -> jnp.dtype:
    """Convert a string representation of a data type to a numpy dtype."""

    if dtype_str not in _dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return _dtype_map[dtype_str]


def is_jax_prng_key(x):
    """Checks if an object is a JAX PRNGKey."""
    # Use the canonical way to check for JAX PRNGKey dtypes
    return isinstance(x, jax.Array) and jax.dtypes.issubdtype(
        x.dtype, jax.dtypes.prng_key
    )


def filter_prng_keys(pytree):
    """Recursively filters out JAX PRNGKeys from a pytree, replacing them with None."""
    return jax.tree.map(
        lambda x: None if is_jax_prng_key(x) else x,
        pytree,
        is_leaf=is_jax_prng_key,  # Treat keys as leaves to prevent recursion into them
    )


@dataclass(unsafe_hash=True, order=True)
class ParamsStats:
    millions: float
    billions: float

    def __repr__(self) -> str:
        return f"ModelParameters(millions={self.millions}, billions={self.billions})"

    def __str__(self) -> str:
        return self.__repr__()


def count_parameters(module: nnx.Module):
    params = nnx.state(module, nnx.Param)
    leaves, _ = jtu.flatten(params)
    sizes = jtu.map(lambda leaf: leaf.size, leaves)
    total = sum(sizes)

    return ParamsStats(
        millions=round(total / 1e6, 2),
        billions=round(total / 1e9, 2),
    )


def load_checkpoint_state(
    model: nnx.Module,
    checkpoint_path: str | Path,
) -> nnx.Module:
    """Load a model from a checkpoint."""
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    print("Created abstract state")

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path).absolute()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, abstract_state)
    # nnx.replace_by_pure_dict(abstract_state, restored_state)
    merged_model = nnx.merge(graphdef, restored_state)
    print("Merged state with the model.")
    return merged_model


def load_sharded_checkpoint_state(
    model: nnx.Module,
    checkpoint_path: str | Path,
    mesh,
) -> nnx.Module:
    """Load a model from a checkpoint."""
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)

    abstract_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
        abstract_state,
        nnx.get_named_sharding(abstract_state, mesh),
    )

    print("Created abstract state")

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path).absolute()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")

    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, abstract_state)
    # nnx.replace_by_pure_dict(abstract_state, restored_state)
    merged_model = nnx.merge(graphdef, restored_state)
    print("Merged state with the model.")
    return merged_model


def checkpoint_post_eval(
    logger: logging.Logger,
    model: nnx.Module,
    checkpoint_dir: Path,
    options: ocp.CheckpointManagerOptions,
    metrics: dict,
    global_step: int,
    epoch: int,
):
    logger.info(
        f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step})..."
    )

    state = nnx.state(model, nnx.Param)
    with ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    ) as mngr:
        mngr.save(
            global_step,
            args=ocp.args.StandardSave(state),
            metrics=metrics,
        )

    logger.info(f"Checkpoint saved at end of epoch {epoch + 1}")
