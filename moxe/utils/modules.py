import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.tree_util as jtu
import orbax.checkpoint as ocp
from flax import nnx

from moxe.tensorboard import TensorBoardLogger


class Sequential(nnx.Module):
    def __init__(self, modules: list[nnx.Module]):
        self.modules = modules
        self._num_modules = len(modules)

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x

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

    def __call__(self, index: Any, *args):
        return jax.lax.switch(
            index,
            self.modules,
            *args,
        )


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


def checkpoint_post_eval(
    logger: logging.Logger,
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    tb_logger: TensorBoardLogger,
    best_metric_key: str,
    checkpoint_manager: ocp.CheckpointManager,
    global_step: int,
    epoch: int,
):
    computed_eval_metrics = metrics.compute()
    logger.info(f"Computed eval metrics: {computed_eval_metrics}")

    # Log evaluation metrics to TensorBoard
    for metric, value in computed_eval_metrics.items():
        tb_logger.log_scalar(f"eval/{metric}", value, global_step)

        # Log evaluation results
    eval_log_data = {
        f"eval_{k}": f"{v.item():.6f}" for k, v in computed_eval_metrics.items()
    }
    logger.info(f"Epoch {epoch + 1} Evaluation Results: {eval_log_data}")

    # Update metrics for checkpointing and save checkpoint
    latest_eval_metrics_for_ckpt = {
        best_metric_key: float(computed_eval_metrics[best_metric_key])
    }

    logger.info(
        f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step}) with eval_loss={latest_eval_metrics_for_ckpt[best_metric_key]:.6f}..."
    )

    state = nnx.state(model, nnx.Param)
    checkpoint_manager.save(
        global_step,
        args=ocp.args.PyTreeSave(state),
        metrics=latest_eval_metrics_for_ckpt,
    )
    checkpoint_manager.wait_until_finished()
    logger.info(f"Checkpoint saved at end of epoch {epoch + 1}")
