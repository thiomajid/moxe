import functools
import json
import logging
import random
import shutil
import time
import typing as tp
from dataclasses import asdict
from pathlib import Path

import grain.python as grain
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import nnx
from huggingface_hub import create_repo, repo_exists, snapshot_download, upload_folder
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from moxe.config import MoxEConfig

# from moxe.inference import GenerationCarry, generate_sequence_scan  # Unused imports
from moxe.modules.model import MoxEForCausalLM
from moxe.output import ConditionedGateOutput, MoxEForwardPassOutput, MoxELayerOutput
from moxe.tensorboard import TensorBoardLogger
from moxe.training.arguments import CustomArgs
from moxe.training.data import DataCollatatorTransform, create_dataloaders
from moxe.training.helpers import apply_gradients, compute_training_steps
from moxe.utils.array import create_mesh, log_node_devices_stats
from moxe.utils.modules import (
    checkpoint_post_eval,
    count_parameters,
    load_sharded_checkpoint_state,
)
from xlstm_jax.utils import str2dtype


def _accumulate_loss(
    outputs: tp.List[MoxELayerOutput | ConditionedGateOutput],
    attr: str,
    is_leaf: tp.Callable[[tp.Any], bool] = None,
):
    """
    Accumulates a specific loss attribute from a Pytree of layer outputs
    in a way that is compatible with JAX tracing.
    """

    def _get_loss_leaf(path, leaf):
        # The leaf could be None from an Optional field, return a scalar zero.
        if leaf is None:
            return 0.0

        # Check if the final part of the path is a GetAttrKey with the desired attribute name.
        # path is a tuple like (TupleKey(0), GetAttrKey('conditioned_output'), GetAttrKey('d_loss'))
        if path and isinstance(path[-1], jtu.GetAttrKey) and path[-1].name == attr:
            return leaf
        else:
            # Return a zero with the same shape and type as the leaf to maintain tree structure.
            return jtu.tree_map(jnp.zeros_like, leaf)

    # Create a new tree containing only the target losses and zeros everywhere else.
    loss_tree = jtu.tree_map_with_path(f=_get_loss_leaf, tree=outputs, is_leaf=is_leaf)

    # Sum all leaves in the newly created, filtered tree.
    total_loss = jtu.tree_reduce(
        jnp.add,
        tree=loss_tree,
        initializer=jnp.zeros(()),
    )

    return total_loss


def loss_fn(
    model: MoxEForCausalLM,
    batch: tuple[jax.Array, jax.Array],
    z_loss_coef: float,
    load_balancing_loss_coef: float,
    d_loss_coef: float = 0.0,
    group_loss_coef: float = 0.0,
):
    """Compute the loss for a batch of data including auxiliary MoE losses."""
    input_ids, labels = batch
    output = model(
        input_ids,
        compute_d_loss=d_loss_coef > 0.0,
        compute_group_loss=group_loss_coef > 0.0,
        return_layers_outputs=True,
    )

    # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
    shifted_logits = rearrange(output.logits[..., :-1, :], "b s v -> (b s) v")

    # shape: [batch, seq] -> [batch * (seq-1)]
    shifted_labels = rearrange(labels[..., 1:], "b s -> (b s)")

    # Compute cross-entropy loss
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits, labels=shifted_labels
    ).mean()

    _dtype = ce_loss.dtype
    total_loss = ce_loss

    # Collect auxiliary losses
    layers_outputs = output.layers_outputs
    num_layers = model.moe.num_layers

    z_loss = _accumulate_loss(layers_outputs, "z_loss")
    z_loss = (z_loss / num_layers).astype(_dtype)

    load_balancing_loss = _accumulate_loss(layers_outputs, "load_balancing_loss")
    load_balancing_loss = (load_balancing_loss / num_layers).astype(_dtype)

    # d-loss
    d_loss: jax.Array = jax.lax.cond(
        d_loss_coef > 0.0,
        lambda: _accumulate_loss(
            layers_outputs,
            "d_loss",
            is_leaf=lambda node: node is not None
            and isinstance(node, ConditionedGateOutput),
        ).astype(_dtype),
        lambda: jnp.zeros((), dtype=_dtype),
    )

    d_loss = d_loss / num_layers

    # group-loss
    group_loss: jax.Array = jax.lax.cond(
        group_loss_coef > 0.0,
        lambda: _accumulate_loss(
            layers_outputs,
            "group_loss",
            is_leaf=lambda node: node is not None
            and isinstance(node, ConditionedGateOutput),
        ).astype(_dtype),
        lambda: jnp.zeros((), dtype=_dtype),
    )

    group_loss = group_loss / num_layers

    # finalize
    total_loss = (
        total_loss
        + z_loss_coef * z_loss
        + load_balancing_loss_coef * load_balancing_loss
        + d_loss * d_loss_coef
        + group_loss * group_loss_coef
    )

    aux_data = MoxEForwardPassOutput(
        ce_loss=ce_loss,
        z_loss=z_loss,
        load_balancing_loss=load_balancing_loss,
        d_loss=d_loss,
        group_loss=group_loss,
    )

    return total_loss, aux_data


@functools.partial(
    nnx.jit,
    static_argnames=(
        "z_loss_coef",
        "load_balancing_loss_coef",
        "d_loss_coef",
        "group_loss_coef",
    ),
)
def compute_grads_and_metrics(
    model: MoxEForCausalLM,
    metrics: nnx.MultiMetric,
    batch: tuple[jax.Array, jax.Array],
    z_loss_coef: float,
    load_balancing_loss_coef: float,
    d_loss_coef: float = 0.0,
    group_loss_coef: float = 0.0,
):
    """Computes gradients, loss, and updates metrics for a single micro-batch."""

    def _loss_fn(model, batch):
        total_loss, aux_losses = loss_fn(
            model,
            batch,
            z_loss_coef,
            load_balancing_loss_coef,
            d_loss_coef,
            group_loss_coef,
        )
        return total_loss, aux_losses

    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux_losses), grads = grad_fn(model, batch)

    perplexity = jnp.exp(aux_losses.ce_loss)
    grad_norm = optax.global_norm(grads)

    metrics.update(
        loss=loss,
        perplexity=perplexity,
        grad_norm=grad_norm,
        ce_loss=aux_losses.ce_loss,
        z_loss=aux_losses.z_loss,
        load_balancing_loss=aux_losses.load_balancing_loss,
        d_loss=aux_losses.d_loss,
        group_loss=aux_losses.group_loss,
    )

    return loss, grads, grad_norm


@functools.partial(
    nnx.jit,
    static_argnames=(
        "z_loss_coef",
        "load_balancing_loss_coef",
        "d_loss_coef",
        "group_loss_coef",
    ),
)
def eval_step(
    model: MoxEForCausalLM,
    metrics: nnx.MultiMetric,
    batch: tuple[jax.Array, jax.Array],
    z_loss_coef: float,
    load_balancing_loss_coef: float,
    d_loss_coef: float = 0.0,
    group_loss_coef: float = 0.0,
):
    """Perform a single evaluation step."""
    total_loss, aux_losses = loss_fn(
        model,
        batch,
        z_loss_coef,
        load_balancing_loss_coef,
        d_loss_coef,
        group_loss_coef,
    )

    perplexity = jnp.exp(aux_losses.ce_loss)

    metrics.update(
        loss=total_loss,
        perplexity=perplexity,
        ce_loss=aux_losses.ce_loss,
        z_loss=aux_losses.z_loss,
        load_balancing_loss=aux_losses.load_balancing_loss,
        d_loss=aux_losses.d_loss,
        group_loss=aux_losses.group_loss,
    )


@functools.partial(
    nnx.jit,
    static_argnames=("config_fn", "seed", "mesh", "dtype"),
)
def _create_sharded_model(
    config_fn: callable,
    seed: int,
    mesh: Mesh,
    dtype=jnp.float32,
):
    rngs = nnx.Rngs(seed)
    config = config_fn()
    model = MoxEForCausalLM(config, mesh=mesh, rngs=rngs, dtype=dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)

    return model


@hydra.main(
    config_path="./configs", config_name="train_moxe_config", version_base="1.1"
)
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting MoxE language model training...")

    parser = HfArgumentParser(CustomArgs)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
    args = tp.cast(CustomArgs, args)

    # Set default warmup_ratio if not provided
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.2
        logger.warning(
            f"warmup_ratio not found in config, defaulting to {args.warmup_ratio}"
        )

    logger.info("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"
        logger.warning("Changed the tokenizer's padding_side from right to left")

    config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config_dict["xlstm"]["vocab_size"] = tokenizer.vocab_size
    config_dict["xlstm"]["pad_token_id"] = tokenizer.pad_token_id
    print("Model config:")
    print(config_dict)

    config = MoxEConfig.from_dict(config_dict)

    log_node_devices_stats(logger)

    # Model instance
    dtype_str = cfg["dtype"]
    logger.info(f"Creating MoxE model with dtype={dtype_str}...")
    dtype = str2dtype(dtype_str)

    mesh_shape = tuple(args.mesh_shape) if hasattr(args, "mesh_shape") else (1,)
    axis_names = tuple(args.axis_names) if hasattr(args, "axis_names") else ("dp",)
    mesh = create_mesh(mesh_shape=mesh_shape, axis_names=axis_names)

    model = None
    with mesh:
        model = _create_sharded_model(
            config_fn=lambda: config,
            seed=args.seed,
            mesh=mesh,
            dtype=dtype,
        )

    logger.info(f"Model parameters: {count_parameters(model)}")
    logger.info(f"Embedding size {count_parameters(model.moe.token_embedding)}")
    logger.info(f"MoE layer size {count_parameters(model.moe.layers[0])}")

    log_node_devices_stats(logger)

    # Handle checkpoint resumption if configured
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

    train_data_ops = [
        grain.Batch(args.per_device_train_batch_size, drop_remainder=True),
        DataCollatatorTransform(
            target_columns=["input_ids", "labels", "attention_mask"],
            collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="np",
            ),
        ),
    ]

    eval_data_ops = [
        grain.Batch(args.per_device_eval_batch_size, drop_remainder=True),
        DataCollatatorTransform(
            target_columns=["input_ids", "labels", "attention_mask"],
            collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="np",
            ),
        ),
    ]

    train_loader, eval_loader = create_dataloaders(
        logger=logger,
        args=args,
        tokenizer=tokenizer,
        max_seq_length=config.xlstm.context_length,
        train_data_ops=train_data_ops,
        eval_data_ops=eval_data_ops,
    )

    # Setup the training loop
    num_train_samples = len(train_loader._data_source)
    num_eval_samples = len(eval_loader._data_source)

    logger.info(f"Dataset sizes - Train: {num_train_samples}, Eval: {num_eval_samples}")
    steps_dict = compute_training_steps(
        args, num_train_samples, num_eval_samples, logger
    )
    max_steps = steps_dict["max_steps"]
    max_optimizer_steps = steps_dict["max_optimizer_steps"]

    # Use optimizer steps for learning rate schedule
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

    optimizer = nnx.Optimizer(model, optimizer_def)

    # Metrics
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        ce_loss=nnx.metrics.Average("ce_loss"),
        z_loss=nnx.metrics.Average("z_loss"),
        load_balancing_loss=nnx.metrics.Average("load_balancing_loss"),
        d_loss=nnx.metrics.Average("d_loss"),
        group_loss=nnx.metrics.Average("group_loss"),
    )

    eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
        ce_loss=nnx.metrics.Average("ce_loss"),
        z_loss=nnx.metrics.Average("z_loss"),
        load_balancing_loss=nnx.metrics.Average("load_balancing_loss"),
        d_loss=nnx.metrics.Average("d_loss"),
        group_loss=nnx.metrics.Average("group_loss"),
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir=args.logging_dir, name="train")

    # Checkpoint manager
    ckpt_dir = Path(args.logging_dir).absolute()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    BEST_METRIC_KEY = "eval_perplexity"
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda metrics: metrics[BEST_METRIC_KEY],
        best_mode="min",
        create=True,
    )

    manager = ocp.CheckpointManager(
        ckpt_dir,
        checkpointers=ocp.PyTreeCheckpointer(),
        options=options,
    )

    # Training loop setup
    global_step = 0
    global_optimizer_step = 0
    latest_eval_metrics_for_ckpt = {BEST_METRIC_KEY: float("inf")}

    logger.info("Starting training loop...")
    logger.info(f"Num Epochs = {args.num_train_epochs}")
    logger.info(f"Micro Batch size = {args.per_device_train_batch_size}")
    logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"Effective Batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    logger.info(
        f"Total batches per epoch: Train - {steps_dict['train_batches']} && Eval - {steps_dict['eval_batches']}"
    )
    logger.info(f"Total steps = {max_steps}")
    logger.info(f"Total optimizer steps = {max_optimizer_steps}")
    logger.info(f"Z-loss coefficient = {args.z_loss_coef}")
    logger.info(f"Load balancing loss coefficient = {args.load_balancing_loss_coef}")
    logger.info(f"D-loss coefficient = {args.d_loss_coef}")
    logger.info(f"Group loss coefficient = {args.group_loss_coef}")

    # Start timing
    training_start_time = time.perf_counter()
    epoch_durations = []

    # Training Loop
    DATA_SHARDING = NamedSharding(
        create_mesh(mesh_shape, axis_names),
        spec=PartitionSpec("dp", None),
    )

    GENERATION_SAMPLES = ["Once upon a time", "There was a girl", "Next to the tree"]
    MAX_NEW_TOKENS = 300
    GREEDY = False
    TEMPERATURE = 0.85

    for epoch in range(args.num_train_epochs):
        epoch_start_time = time.perf_counter()
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
        train_metrics.reset()

        epoch_desc = f"Epoch {epoch + 1}/{args.num_train_epochs}"
        with tqdm(
            total=steps_dict["steps_per_epoch"],
            desc=epoch_desc,
            leave=True,
        ) as pbar:
            pbar.set_description(epoch_desc)
            for step, batch in enumerate(train_loader):
                global_step += 1  # Count every batch as a step

                # Prepare batch - Squeeze on axis 0 because Grain creates an additional axis
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"], dtype=jnp.int32)

                if input_ids.ndim == 3 and input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)

                if labels.ndim == 3 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)

                # Apply data sharding
                input_ids = jax.device_put(input_ids, DATA_SHARDING)
                labels = jax.device_put(labels, DATA_SHARDING)

                _batch = (input_ids, labels)

                # Compute gradients and metrics
                loss, grads, grad_norm = compute_grads_and_metrics(
                    model=model,
                    metrics=train_metrics,
                    batch=_batch,
                    z_loss_coef=args.z_loss_coef,
                    load_balancing_loss_coef=args.load_balancing_loss_coef,
                    d_loss_coef=args.d_loss_coef,
                    group_loss_coef=args.group_loss_coef,
                )

                apply_gradients(optimizer, grads)

                # Check if it's time for optimizer step
                is_update_step = (step + 1) % args.gradient_accumulation_steps == 0
                if is_update_step:
                    global_optimizer_step += 1

                    # Log learning rate
                    current_lr = cosine_schedule(global_optimizer_step)
                    tb_logger.log_learning_rate(current_lr, global_optimizer_step)

                # Logging
                if global_step % args.logging_steps == 0:
                    computed_metrics = train_metrics.compute()

                    # Log metrics to TensorBoard
                    for metric, value in computed_metrics.items():
                        tb_logger.log_scalar(f"train/{metric}", value, global_step)

                    train_metrics.reset()

                # Update progress bar
                current_lr = cosine_schedule(global_optimizer_step)
                postfix_data = {
                    "step": f"{global_step}/{max_steps}",
                    "opt_step": f"{global_optimizer_step}/{max_optimizer_steps}",
                    "lr": f"{current_lr:.2e}",
                    "loss": f"{loss.item():.6f}",
                    "grad_norm": f"{grad_norm.item():.4f}",
                }

                # Add best metrics
                if (
                    BEST_METRIC_KEY in latest_eval_metrics_for_ckpt
                    and latest_eval_metrics_for_ckpt[BEST_METRIC_KEY] != float("inf")
                ):
                    postfix_data["best_loss"] = (
                        f"{latest_eval_metrics_for_ckpt[BEST_METRIC_KEY]:.6f}"
                    )

                current_desc = f"Epoch {epoch + 1}/{args.num_train_epochs} (Step {global_step}/{max_steps}, Opt {global_optimizer_step}/{max_optimizer_steps})"
                pbar.set_description(current_desc)
                pbar.set_postfix(postfix_data)
                pbar.update(1)

            # --- Evaluation after each epoch ---

        eval_start_time = time.perf_counter()
        logger.info(f"Starting evaluation after epoch {epoch + 1}...")
        eval_metrics.reset()

        eval_batch_count = 0
        eval_data_available = True

        try:
            for batch in tqdm(
                eval_loader,
                desc=f"Evaluating Epoch {epoch + 1}",
                leave=False,
            ):
                eval_batch_count += 1
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"], dtype=jnp.int32)

                if input_ids.ndim == 3 and input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)

                if labels.ndim == 3 and labels.shape[0] == 1:
                    labels = labels.squeeze(0)

                # Apply data sharding
                input_ids = jax.device_put(input_ids, DATA_SHARDING)
                labels = jax.device_put(labels, DATA_SHARDING)
                _batch = (input_ids, labels)

                eval_step(
                    model=model,
                    metrics=eval_metrics,
                    batch=_batch,
                    z_loss_coef=args.z_loss_coef,
                    load_balancing_loss_coef=args.load_balancing_loss_coef,
                    d_loss_coef=args.d_loss_coef,
                    group_loss_coef=args.group_loss_coef,
                )
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            eval_data_available = False

        logger.info(f"Processed {eval_batch_count} evaluation batches")

        if eval_batch_count > 0 and eval_data_available:
            checkpoint_post_eval(
                logger=logger,
                model=model,
                metrics=eval_metrics,
                tb_logger=tb_logger,
                best_metric_key=BEST_METRIC_KEY,
                checkpoint_manager=manager,
                global_step=global_step,
                epoch=epoch,
            )

            # Generate some text
            choosen_prompt = random.choice(GENERATION_SAMPLES)
            input_ids = tokenizer(choosen_prompt, return_tensors="jax", padding=True)[
                "input_ids"
            ]

            # Simple greedy generation for monitoring
            try:
                with jax.disable_jit():
                    for _ in range(min(MAX_NEW_TOKENS, 50)):  # Limit for monitoring
                        output = model(input_ids)
                        next_token_logits = output.logits[0, -1, :]
                        if GREEDY:
                            next_token = jnp.argmax(next_token_logits)
                        else:
                            next_token = jax.random.categorical(
                                jax.random.PRNGKey(42 + epoch),
                                next_token_logits / TEMPERATURE,
                            )
                        input_ids = jnp.concatenate(
                            [input_ids, next_token[None, None]], axis=1
                        )

                        # Stop if EOS token
                        if next_token == tokenizer.eos_token_id:
                            break

                generated_text = tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                )
                tb_logger.writer.text(
                    "train/generation", generated_text, step=global_step
                )
                logger.info(f"Generated: {generated_text}")
            except Exception as e:
                logger.warning(f"Text generation failed: {e}")

        else:
            logger.warning(
                f"No evaluation data processed for epoch {epoch + 1}. Eval loader might be empty or misconfigured."
            )

        # Record evaluation duration and log to TensorBoard
        eval_end_time = time.perf_counter()
        eval_duration = eval_end_time - eval_start_time
        tb_logger.log_scalar("timing/eval_duration", eval_duration, global_step)
        logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")

        # Record epoch duration and log to TensorBoard
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        tb_logger.log_scalar("timing/epoch_duration", epoch_duration, global_step)
        logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    logger.info("Training completed.")

    # Log final training metrics
    if train_metrics is not None:
        try:
            final_computed_metrics = train_metrics.compute()
            if final_computed_metrics and any(
                v.item() != 0 for v in final_computed_metrics.values()
            ):
                logger.info(
                    "Logging final training metrics from last accumulation cycle..."
                )

                key_metric = "loss"
                if key_metric in final_computed_metrics:
                    latest_eval_metrics_for_ckpt = {
                        BEST_METRIC_KEY: float(final_computed_metrics[key_metric])
                    }

                # Log final metrics to TensorBoard
                for metric, value in final_computed_metrics.items():
                    tb_logger.log_scalar(f"train/{metric}", value, global_step)

                current_lr = cosine_schedule(global_optimizer_step)
                tb_logger.log_learning_rate(current_lr, global_optimizer_step)
                logger.info(f"Final learning rate: {current_lr:.2e}")

                # Log final auxiliary loss breakdown
                logger.info("Final Training Metrics:")
                logger.info(f"  Total Loss: {final_computed_metrics['loss']:.6f}")
                logger.info(f"  CE Loss: {final_computed_metrics['ce_loss']:.6f}")
                logger.info(f"  Z Loss: {final_computed_metrics['z_loss']:.6f}")
                logger.info(
                    f"  Load Balancing Loss: {final_computed_metrics['load_balancing_loss']:.6f}"
                )
                logger.info(f"  D Loss: {final_computed_metrics['d_loss']:.6f}")
                logger.info(f"  Group Loss: {final_computed_metrics['group_loss']:.6f}")
                logger.info(f"  Perplexity: {final_computed_metrics['perplexity']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute final training metrics: {e}")

    # Calculate total training duration and log to TensorBoard
    training_end_time = time.perf_counter()
    total_training_duration = training_end_time - training_start_time
    tb_logger.log_scalar(
        "timing/total_training_duration", total_training_duration, global_step
    )

    # Calculate and log timing statistics
    avg_epoch_duration = (
        sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0
    )

    tb_logger.log_scalar("timing/avg_epoch_duration", avg_epoch_duration, global_step)

    logger.info(
        f"Training completed in {total_training_duration:.2f} seconds ({total_training_duration / 3600:.2f} hours)"
    )
    logger.info(f"Average epoch duration: {avg_epoch_duration:.2f} seconds")

    # Close TensorBoard logger
    tb_logger.close()

    # Final saving and upload
    logger.info("Saving final artifacts...")
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copy TensorBoard logs to artifacts directory
    tb_logs_source = Path(args.logging_dir) / "training"
    tb_logs_target = artifacts_dir / "tensorboard_logs"
    if tb_logs_source.exists():
        shutil.copytree(tb_logs_source, tb_logs_target, dirs_exist_ok=True)
        logger.info(f"TensorBoard logs copied to {tb_logs_target}")

    # Save training history (keeping minimal data for compatibility)
    training_summary = {
        "total_training_duration": total_training_duration,
        "avg_epoch_duration": avg_epoch_duration,
        "num_epochs_completed": len(epoch_durations),
        "global_steps": global_step,
        "global_optimizer_steps": global_optimizer_step,
        "auxiliary_loss_coefficients": {
            "z_loss_coef": args.z_loss_coef,
            "load_balancing_loss_coef": args.load_balancing_loss_coef,
            "d_loss_coef": args.d_loss_coef,
            "group_loss_coef": args.group_loss_coef,
        },
    }
    with open(artifacts_dir / "train_history.json", "w") as f:
        json.dump(training_summary, f, indent=4)
    logger.info(f"Training history saved to {artifacts_dir / 'train_history.json'}")

    # Save model config
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    logger.info(f"Model config saved to {artifacts_dir / 'config.json'}")

    # Save trainer config
    with open(artifacts_dir / "trainer_config.json", "w") as f:
        trainer_config_dict = asdict(args)
        if "hub_token" in trainer_config_dict:
            trainer_config_dict.pop("hub_token")
        json.dump(trainer_config_dict, f, indent=4)
    logger.info(f"Trainer config saved to {artifacts_dir / 'trainer_config.json'}")

    # Saving the tokenizer
    tokenizer.save_pretrained(artifacts_dir)

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
    logger.info(f"Timing summary saved to {artifacts_dir / 'timing_summary.json'}")

    # Save final model state
    if global_step > 0:
        final_model_state = nnx.state(model, nnx.Param)
        logger.info(
            f"Saving final model state at step {global_step} to be considered by CheckpointManager with metrics {latest_eval_metrics_for_ckpt}."
        )

        manager.save(
            global_step,
            final_model_state,
            metrics=latest_eval_metrics_for_ckpt,
        )
        manager.wait_until_finished()

    # Copy best checkpoint to artifacts directory
    best_step_to_deploy = manager.best_step()
    target_ckpt_deployment_path = artifacts_dir / "model_checkpoint"

    if best_step_to_deploy is not None:
        logger.info(
            f"Best checkpoint according to CheckpointManager is at step {best_step_to_deploy} (based on {BEST_METRIC_KEY})."
        )
        source_ckpt_dir = manager.directory / str(best_step_to_deploy)

        if source_ckpt_dir.exists():
            logger.info(
                f"Copying best checkpoint from {source_ckpt_dir} to {target_ckpt_deployment_path}"
            )
            if target_ckpt_deployment_path.exists():
                shutil.rmtree(target_ckpt_deployment_path)

            target_ckpt_deployment_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_ckpt_dir, target_ckpt_deployment_path, dirs_exist_ok=True
            )
        else:
            logger.error(f"Best checkpoint directory {source_ckpt_dir} not found.")
    else:
        logger.warning("CheckpointManager did not identify a best checkpoint.")
        if global_step > 0:
            final_model_state = nnx.state(model, nnx.Param)
            logger.info(
                f"Saving current final model state directly to {target_ckpt_deployment_path} as a fallback."
            )
            if target_ckpt_deployment_path.exists():
                shutil.rmtree(target_ckpt_deployment_path)
            target_ckpt_deployment_path.mkdir(parents=True, exist_ok=True)
            ocp.PyTreeCheckpointHandler().save(
                target_ckpt_deployment_path,
                final_model_state,
            )
        else:
            logger.error(
                "No optimizer steps were completed, and no best checkpoint found. Cannot save a model."
            )

    # Push to Hub if requested
    if args.push_to_hub:
        logger.info(
            f"Pushing artifacts from {artifacts_dir} to Hugging Face Hub repository: {args.hub_model_id}..."
        )
        if not repo_exists(args.hub_model_id, token=args.hub_token):
            logger.info(f"Creating repository {args.hub_model_id}...")
            create_repo(
                repo_id=args.hub_model_id,
                token=args.hub_token,
                private=args.hub_private_repo,
                exist_ok=True,
            )

        upload_folder(
            repo_id=args.hub_model_id,
            folder_path=artifacts_dir,
            token=args.hub_token,
            commit_message=cfg.get(
                "upload_message", "MoxE language model training completed"
            ),
        )
        logger.info("Push to Hub completed.")


if __name__ == "__main__":
    main()
