import sys

sys.path.append("../")

import logging
import typing as tp
from functools import partial
from pathlib import Path
from pprint import pprint

import grain.python as grain
import hydra
import jax
import jax.debug
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import checkpoint_managers
from transformers import AutoTokenizer, HfArgumentParser
from transformers.data.data_collator import DataCollatorForLanguageModeling

from training.arguments import TrainingArguments, compute_training_steps
from training.callback import (
    CheckpointCallback,
    GenerateTextCallback,
    PushToHubCallback,
)
from training.data import create_dataloaders
from training.data_transformations import CollateForLanguageModeling
from training.loss import causal_lm_loss
from training.tensorboard import TensorBoardLogger
from training.trainer import Trainer
from training.utils.array import create_mesh, log_node_devices_stats
from training.utils.module import (
    count_parameters,
    load_sharded_checkpoint_state,
    str2dtype,
)
from xlstm_jax import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.parser import parse_xlstm_config_dict


def lm_loss(model: xLSTMLMModel, batch: tuple[jax.Array, jax.Array]):
    input_ids, labels = batch
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    return loss


@nnx.jit
def train_step(
    model: xLSTMLMModel,
    batch: tuple[jax.Array, jax.Array],
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
):
    # with jax.debug_nans(True):
    grad_fn = nnx.value_and_grad(lm_loss)
    loss, grads = grad_fn(model, batch)

    # Debugging NaNs
    # jax.debug.print("loss: {loss}", loss=loss)
    # is_nan_loss = jnp.isnan(loss)
    # jax.debug.print("is_nan_loss: {is_nan_loss}", is_nan_loss=is_nan_loss)

    grad_norm = optax.global_norm(grads)
    # jax.debug.print("grad_norm: {grad_norm}", grad_norm=grad_norm)
    # is_nan_grad = jnp.isnan(grad_norm)
    # jax.debug.print("is_nan_grad: {is_nan_grad}", is_nan_grad=is_nan_grad)

    optimizer.update(model, grads)

    metrics.update(
        loss=loss,
        perplexity=jnp.exp(loss),
        grad_norm=grad_norm,
    )

    return loss, grads, grad_norm


@nnx.jit
def eval_step(
    model: xLSTMLMModel,
    batch: tuple[jax.Array, jax.Array],
    metrics: nnx.MultiMetric,
):
    loss = lm_loss(model, batch)
    metrics.update(loss=loss, perplexity=jnp.exp(loss))
    return loss


@partial(
    nnx.jit,
    static_argnames=("config", "mesh", "dtype", "param_dtype"),
)
def _create_sharded_model(
    config: xLSTMLMModelConfig,
    rngs: nnx.Rngs,
    mesh: Mesh,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
):
    model = xLSTMLMModel(
        config,
        mesh=mesh,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


@hydra.main(
    config_path="../configs",
    config_name="train_causal_lm",
    version_base="1.2",
)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting xLSTMLMModel training...")

    parser = HfArgumentParser(TrainingArguments)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]  # ty: ignore
    args = tp.cast(TrainingArguments, args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=args.hub_token)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"
        logger.warning("Changed the tokenizer's padding_side from right to left")

    # Create model config from cfg
    log_node_devices_stats(logger)
    dtype = str2dtype(args.dtype)
    param_dtype = str2dtype(args.param_dtype)
    logger.info(
        f"Creating xLSTM model with dtype={dtype} and param_dtype={param_dtype}..."
    )

    mesh_shape = tuple(args.mesh_shape)
    axis_names = tuple(args.axis_names)
    mesh = create_mesh(mesh_shape=mesh_shape, axis_names=axis_names)
    rngs = nnx.Rngs(args.seed)

    model_config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    model_config_dict["vocab_size"] = tokenizer.vocab_size
    model_config_dict["pad_token_id"] = tokenizer.pad_token_id

    pprint(model_config_dict)

    config = parse_xlstm_config_dict(model_config_dict)  # ty: ignore

    model: xLSTMLMModel

    with mesh:
        model = _create_sharded_model(
            config,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    logger.info(f"Model parameters: {count_parameters(model)}")
    logger.info(f"Embedding num params: {count_parameters(model.token_embedding)}")
    logger.info(f"xLSTM blocks size {count_parameters(model.xlstm_block_stack)}")

    log_node_devices_stats(logger)

    if cfg.get("resume_from_checkpoint", False):
        logger.info(f"Resuming from checkpoint from {cfg['checkpoint_hub_url']}")
        save_dir = Path(cfg["checkpoint_save_dir"])

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=cfg["checkpoint_hub_url"],
            local_dir=save_dir,
            token=args.hub_token,
            revision=cfg.get("checkpoint_revision", "main"),
        )

        ckpt_path = save_dir / "model_checkpoint/default"
        ckpt_path = ckpt_path.absolute()

        load_sharded_checkpoint_state(
            model=model,
            checkpoint_path=ckpt_path,
            mesh=mesh,
        )

        logger.info("loaded checkpoint state")

    logger.info("Setting model in training mode")
    model.train()

    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    # Create data transforms pipeline
    TARGET_COLUMNS = ["input_ids"]
    train_transforms = [
        grain.Batch(batch_size=args.per_device_train_batch_size, drop_remainder=True),
        CollateForLanguageModeling(
            target_columns=TARGET_COLUMNS,
            collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                return_tensors="np",
                mlm=False,
            ),
        ),
    ]

    eval_transforms = [
        grain.Batch(batch_size=args.per_device_eval_batch_size, drop_remainder=True),
        CollateForLanguageModeling(
            target_columns=TARGET_COLUMNS,
            collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                return_tensors="np",
                mlm=False,
            ),
        ),
    ]

    train_loader, eval_loader = create_dataloaders(
        logger=logger,
        args=args,
        tokenizer=tokenizer,
        max_seq_length=config.context_length,
        target_columns=TARGET_COLUMNS,
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )

    # Setup the training loop
    num_train_samples = len(train_loader._data_source)
    num_eval_samples = len(eval_loader._data_source)

    logger.info(f"Dataset sizes - Train: {num_train_samples}, Eval: {num_eval_samples}")
    train_steps_config = compute_training_steps(
        args,
        num_train_samples,
        num_eval_samples,
        logger,
    )

    max_steps = train_steps_config.max_steps
    max_optimizer_steps = train_steps_config.max_optimizer_steps

    # Set default warmup_ratio if not provided
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.2
        logger.warning(
            f"warmup_ratio not found in config, defaulting to {args.warmup_ratio}"
        )

    # Use optimizer steps for learning rate schedule (not micro-batch steps)
    warmup_steps = int(args.warmup_ratio * max_optimizer_steps)
    logger.info(
        f"Calculated warmup steps: {warmup_steps} ({args.warmup_ratio=}, max_optimizer_steps={max_optimizer_steps})"
    )

    # Create warmup cosine learning rate schedule
    cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=int(max_optimizer_steps - warmup_steps),
        end_value=args.learning_rate * 0.2,
    )

    logger.info(
        f"Using warmup cosine learning rate schedule: 0.0 -> {args.learning_rate} -> {args.learning_rate * 0.2} over {max_optimizer_steps} optimizer steps (warmup: {warmup_steps} steps)"
    )

    # Optimizer
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(
            learning_rate=cosine_schedule,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            weight_decay=args.weight_decay,
        ),
    )

    optimizer_def = optax.MultiSteps(
        optimizer_def,
        every_k_schedule=args.gradient_accumulation_steps,
    )

    optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)  # ty: ignore

    # Metrics
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        perplexity=nnx.metrics.Average("perplexity"),
    )

    eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
    )

    # TensorBoard logger
    reporter = TensorBoardLogger(
        log_dir=args.logging_dir,
        name=args.hub_model_id.split("/")[-1],
    )

    # Checkpoint manager
    checkpoint_dir = Path(args.logging_dir).absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Use evaluation reconstruction loss for best model selection
    checkpoint_options = ocp.CheckpointManagerOptions(
        best_mode="min",
        create=True,
        preservation_policy=checkpoint_managers.BestN(
            get_metric_fn=lambda metrics: metrics[args.best_metric_key],
            n=args.best_n_to_keep,
            keep_checkpoints_without_metrics=False,
        ),
    )

    logger.info("Starting training loop...")
    logger.info(f"Num Epochs = {args.num_train_epochs}")
    logger.info(f"Micro Batch size = {args.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"Effective Batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    logger.info(
        f"Total batches per epoch: Train - {train_steps_config.train_batches} && Eval - {train_steps_config.eval_batches}"
    )
    logger.info(f"Total steps = {max_steps}")
    logger.info(f"Total optimizer steps = {max_optimizer_steps}")

    DATA_SHARDING = NamedSharding(mesh, spec=P("dp", None))
    num_generation_samples = 8
    SAMPLE_TOKENS = next(iter(train_loader))["input_ids"]
    SAMPLE_TOKENS = jnp.array(SAMPLE_TOKENS)

    if SAMPLE_TOKENS.shape[0] == 1:
        SAMPLE_TOKENS = SAMPLE_TOKENS.squeeze(0)

    CONTEXT_WINDOW = config.context_length
    SAMPLE_TOKENS = SAMPLE_TOKENS[:num_generation_samples, : CONTEXT_WINDOW // 2]
    SAMPLE_TOKENS = jax.device_put(SAMPLE_TOKENS, DATA_SHARDING)

    logger.info(f"Samples tokens shape: {SAMPLE_TOKENS.shape}")

    CALLBACKS = [
        CheckpointCallback(
            model=model,
            options=checkpoint_options,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        ),
        GenerateTextCallback(
            model=model,
            tokenizer=tokenizer,
            sample_tokens=SAMPLE_TOKENS,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["temperature"],
            greedy=cfg["greedy"],
            reporter=reporter,
            logger=logger,
        ),
        PushToHubCallback(logger=logger, args=args),
    ]

    # _train_step_fn = nnx.cached_partial(train_step, model, optimizer)
    # _eval_step_fn = nnx.cached_partial(eval_step, model)

    trainer = Trainer(
        model=model,
        model_config=config,
        args=args,
        optimizer=optimizer,
        lr_scheduler=cosine_schedule,
        mesh=mesh,
        train_step_fn=train_step,
        eval_step_fn=eval_step,
        data_sharding=DATA_SHARDING,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        reporter=reporter,
        logger=logger,
        steps_config=train_steps_config,
        callbacks=CALLBACKS,
    )

    trainer.train()

    logger.info("Everything is done")


if __name__ == "__main__":
    main()
