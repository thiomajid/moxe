from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp
from flax import nnx


class Sequential(nnx.Module):
    def __init__(self, modules: list[nnx.Module]):
        self.modules = modules
        self._num_modules = len(modules)

    def __call__(self, x):
        def _module_scan(carry, idx: jax.Array):
            next_state = jax.lax.switch(idx, self.modules, operand=carry)
            return next_state, None

        out, _ = jax.lax.scan(f=_module_scan, init=x, xs=jnp.arange(self._num_modules))
        return out

    def __len__(self):
        return self._num_modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int):
        return self.modules[idx]


class ModuleList(nnx.Module):
    def __init__(self, modules: list[nnx.Module]):
        self.modules = modules

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx: int):
        return self.modules[idx]


@dataclass
class ParamsStats:
    millions: float
    billions: float

    def __repr__(self) -> str:
        return f"ParamsStats(millions={self.millions}, billions={self.billions})"

    def __str__(self) -> str:
        return self.__repr__()


def count_parameters(module: nnx.Module):
    params = nnx.state(module, nnx.Param)
    leaves, _ = jtu.tree_flatten(params)
    sizes = jtu.tree_map(lambda leaf: leaf.size, leaves)
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
