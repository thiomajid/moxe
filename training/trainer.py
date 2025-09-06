import json
import logging
import shutil
import typing as tp
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import chex
import grain.python as grain
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from tqdm.auto import tqdm

from training.arguments import TrainingArguments, TrainingSteps
from training.callback import Callback
from training.state import TrainerState
from training.tensorboard import TensorBoardLogger
from training.utils.module import count_parameters

_Batch = tuple[jax.Array, jax.Array]  # input_ids and labels

_TrainStepFn = tp.Callable[
    [nnx.Module, _Batch, nnx.Optimizer, nnx.MultiMetric],
    tuple[jax.Array, chex.ArrayTree, jax.Array],
]

_EvalStepFn = tp.Callable[[nnx.Module, _Batch, nnx.MultiMetric], jax.Array]


class Trainer:
    def __init__(
        self,
        model: nnx.Module,
        model_config: tp.Any,
        args: TrainingArguments,
        optimizer: nnx.Optimizer,
        lr_scheduler: optax.Schedule,
        mesh: Mesh,
        train_step_fn: _TrainStepFn,
        eval_step_fn: _EvalStepFn,
        data_sharding: NamedSharding,
        train_dataloader: grain.DataLoader,
        eval_dataloader: grain.DataLoader,
        train_metrics: nnx.Metric,
        eval_metrics: nnx.Metric,
        reporter: TensorBoardLogger,
        logger: logging.Logger,
        steps_config: TrainingSteps,
        callbacks: tp.Optional[list[Callback]] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.reporter = reporter
        self.logger = logger
        self.steps_config = steps_config

        self.state = TrainerState()
        self._train_step_fn: _TrainStepFn = train_step_fn
        self._eval_step_fn: _EvalStepFn = eval_step_fn
        self.callbacks = tuple(callbacks) if callbacks is not None else None

    def on_train_start(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        self.state.epoch += 1
        metrics = self.eval_metrics.compute()

        candidate_best_metric = metrics[self.args.best_metric_key]
        if candidate_best_metric < self.state.best_metric_value:
            self.state.best_metric_value = candidate_best_metric
            self.state.best_checkpoint_step = self.state.current_step

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_epoch_end(self.state, metrics)

    def train(self):
        labels_dtype = jnp.int32
        epoch_durations = []

        self.on_train_start()  #! Callback hook
        training_start_time = perf_counter()

        for epoch in range(self.args.num_train_epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}/{self.args.num_train_epochs}")
            self.train_metrics.reset()

            self.on_epoch_start()  #! Callback hook
            epoch_start_time = perf_counter()
            epoch_desc = f"Epoch {epoch + 1}/{self.args.num_train_epochs}"

            train_pbar = tqdm(
                self.train_dataloader,
                total=self.steps_config.steps_per_epoch,
                desc=epoch_desc,
                leave=True,
            )

            train_pbar.set_description(epoch_desc)
            self.model.train()

            for batch in train_pbar:
                self.state.current_step += 1  # Count every batch as a step
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"], dtype=labels_dtype)

                # Grain may add an additional batch dim
                if input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)

                if labels.shape[0] == 1:
                    labels = labels.squeeze(0)

                # Data placement
                input_ids = jax.device_put(input_ids, self.data_sharding)
                labels = jax.device_put(labels, self.data_sharding)
                step_batch = (input_ids, labels)

                loss, grads, grad_norm = self._train_step_fn(
                    self.model,
                    step_batch,
                    self.optimizer,
                    self.train_metrics,
                )

                postfix_data = {}

                # Check if it's time for optimizer step
                is_update_step = (
                    self.state.current_step % self.args.gradient_accumulation_steps == 0
                )

                if is_update_step:
                    self.state.optimizer_step += 1

                    # Log learning rate
                    current_lr = self.lr_scheduler(self.state.optimizer_step)
                    self.reporter.log_learning_rate(
                        current_lr, self.state.optimizer_step
                    )

                    postfix_data["lr"] = f"{current_lr:.2e}"
                    train_pbar.set_postfix(postfix_data)

                # Logging
                if self.state.current_step % self.args.logging_steps == 0:
                    computed_metrics = self.train_metrics.compute()
                    for metric, value in computed_metrics.items():
                        self.reporter.log_scalar(
                            f"train/{metric}",
                            value,
                            self.state.current_step,
                        )

                    self.train_metrics.reset()

                    postfix_data = {
                        "step": f"{self.state.current_step}/{self.steps_config.max_steps}",
                        "opt_step": f"{self.state.optimizer_step}/{self.steps_config.max_optimizer_steps}",
                        "loss": f"{loss.item():.6f}",
                        "grad_norm": f"{grad_norm.item():.4f}",
                    }
                    train_pbar.set_postfix(postfix_data)

                current_desc = f"Epoch {epoch + 1}/{self.args.num_train_epochs} (Step {self.state.current_step}/{self.steps_config.max_steps}, Opt {self.state.optimizer_step}/{self.steps_config.max_optimizer_steps})"
                train_pbar.set_description(current_desc)
                # train_pbar.update(1)

            #! Evaluation after each epoch
            self.logger.info(f"Starting evaluation after epoch {epoch + 1}...")
            self.eval_metrics.reset()
            eval_batch_count = 0

            self.model.eval()
            eval_start_time = perf_counter()
            eval_pbar = tqdm(
                self.eval_dataloader,
                desc=f"Evaluating Epoch {epoch + 1}",
                leave=False,
            )

            for batch in eval_pbar:
                eval_batch_count += 1
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"], dtype=labels_dtype)

                # Grain may add an additional batch dim
                if input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)

                if labels.shape[0] == 1:
                    labels = labels.squeeze(0)

                # Data placement
                input_ids = jax.device_put(input_ids, self.data_sharding)
                labels = jax.device_put(labels, self.data_sharding)
                step_batch = (input_ids, labels)
                self._eval_step_fn(self.model, step_batch, self.eval_metrics)

                self.logger.info(f"Processed {eval_batch_count} evaluation batches")
                eval_end_time = perf_counter()
                eval_duration = eval_end_time - eval_start_time

                self.reporter.log_scalar(
                    "timing/eval_duration",
                    eval_duration,
                    self.state.current_step,
                )

                self.logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")

            # Record epoch duration and log to TensorBoard
            epoch_end_time = perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_durations.append(epoch_duration)
            self.reporter.log_scalar(
                "timing/epoch_duration", epoch_duration, self.state.current_step
            )

            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds"
            )

            computed_eval_metrics = self.eval_metrics.compute()
            for metric, value in computed_eval_metrics.items():
                self.reporter.log_scalar(
                    f"eval/{metric}",
                    value,
                    self.state.current_step,
                )

            self.logger.info(
                f"Epoch {epoch + 1} Evaluation Results: {computed_eval_metrics}"
            )

            self.on_epoch_end()  #! callback hook

        training_end_time = perf_counter()
        self.logger.info("Training completed")

        self.on_train_end(  #! callback hook
            epoch_durations=epoch_durations,
            training_start_time=training_start_time,
            training_end_time=training_end_time,
        )

    def on_train_end(
        self,
        training_start_time: float,
        training_end_time: float,
        epoch_durations: list[float],
    ):
        total_training_duration = training_end_time - training_start_time
        self.reporter.log_scalar(
            "timing/total_training_duration",
            total_training_duration,
            self.state.current_step,
        )

        # Calculate and log timing statistics
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
        self.reporter.log_scalar(
            "timing/avg_epoch_duration",
            avg_epoch_duration,
            self.state.current_step,
        )

        self.logger.info(
            f"Training completed in {total_training_duration:.2f} seconds ({total_training_duration / 3600:.2f} hours)"
        )
        self.logger.info(f"Average epoch duration: {avg_epoch_duration:.2f} seconds")

        # Close TensorBoard logger
        self.reporter.close()

        # Final saving and upload
        self.logger.info("Saving final artifacts...")
        artifacts_dir = Path(self.args.output_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copy TensorBoard logs to artifacts directory
        tb_logs_source = Path(self.args.logging_dir) / "training"
        tb_logs_target = artifacts_dir / "tensorboard_logs"
        if tb_logs_source.exists():
            shutil.copytree(tb_logs_source, tb_logs_target, dirs_exist_ok=True)
            self.logger.info(f"TensorBoard logs copied to {tb_logs_target}")

        # Save training history (keeping minimal data for compatibility)
        training_summary = {
            "total_training_duration": total_training_duration,
            "avg_epoch_duration": avg_epoch_duration,
            "num_epochs_completed": len(epoch_durations),
            "global_steps": self.state.current_step,
            "global_optimizer_steps": self.state.optimizer_step,
            "params": asdict(count_parameters(self.model)),
        }
        with open(artifacts_dir / "train_history.json", "w") as f:
            json.dump(training_summary, f, indent=4)
        self.logger.info(
            f"Training history saved to {artifacts_dir / 'train_history.json'}"
        )

        with open(artifacts_dir / "config.json", "w") as f:
            json.dump(asdict(self.model_config), f, indent=4)
        self.logger.info(f"Model config saved to {artifacts_dir / 'config.json'}")

        # Save trainer config
        with open(artifacts_dir / "trainer_config.json", "w") as f:
            trainer_config_dict = asdict(self.args)
            if "hub_token" in trainer_config_dict:
                trainer_config_dict.pop("hub_token")
            json.dump(trainer_config_dict, f, indent=4)
        self.logger.info(
            f"Trainer config saved to {artifacts_dir / 'trainer_config.json'}"
        )

        # Save timing summary
        timing_summary = {
            "total_training_duration_seconds": total_training_duration,
            "total_training_duration_hours": total_training_duration / 3600,
            "average_epoch_duration_seconds": avg_epoch_duration,
            "num_epochs_completed": len(epoch_durations),
            "num_evaluations_completed": len(epoch_durations),  # One eval per epoch
        }
        with open(artifacts_dir / "timing_summary.json", "w") as f:
            json.dump(timing_summary, f, indent=4)
        self.logger.info(
            f"Timing summary saved to {artifacts_dir / 'timing_summary.json'}"
        )

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_end()
