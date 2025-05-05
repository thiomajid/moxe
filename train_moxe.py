import functools
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import cast

import hydra
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import nnx
from huggingface_hub import create_repo, repo_exists, upload_folder
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)

from moxe import MoxEForCausalLM
from moxe._trainer.arguments import CustomArgs
from moxe.output import MoxEForwardPassOutput
from xlstm_jax.utils import str2dtype


def loss_fn(
    model: MoxEForCausalLM,
    batch: tuple[jnp.ndarray, ...],
    z_loss_coef: float,
    load_balancing_loss_coef: float,
    d_loss_coef: float = 0.0,
    group_loss_coef: float = 0.0,
):
    """Compute the loss for a batch of data."""
    input_ids, labels = batch
    output = model(input_ids, return_layers_outputs=True)

    # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
    shifted_logits = rearrange(output.logits[..., :-1, :], "b s v -> (b s) v")

    # shape: [batch, seq] -> [batch * (seq-1)]
    shifted_labels = rearrange(labels[..., 1:], "b s -> (b s)")

    # Compute cross-entropy loss
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=shifted_logits, labels=shifted_labels
    ).mean()

    total_loss = ce_loss

    # -------------- Z-loss computation --------------------------------

    per_layer_z_loss_tree = jtu.tree_map(
        f=lambda layer: layer.z_loss,
        tree=output.layers_outputs,
    )

    # accumulate z_loss per-layer
    z_loss = jtu.tree_reduce(
        jnp.add,
        tree=per_layer_z_loss_tree,
        initializer=jnp.zeros(()),
    ).mean()

    # -------------- LOAD BALANCING LOSS computation ---------------------
    per_layer_load_balancing_loss = jtu.tree_map(
        f=lambda layer: layer.load_balancing_loss,
        tree=output.layers_outputs,
    )

    # reduction step
    load_balancing_loss = jtu.tree_reduce(
        jnp.add,
        tree=per_layer_load_balancing_loss,
        initializer=jnp.zeros(()),
    ).mean()

    # ---------------------- DIFFICULTY LOSS ------------------------
    per_layer_d_loss = jtu.tree_map(
        f=lambda layer: layer.conditioned_output.d_loss,
        tree=output.layers_outputs,
    )

    # reduction step
    d_loss = jtu.tree_reduce(
        jnp.add,
        tree=per_layer_d_loss,
        initializer=jnp.zeros(()),
    )

    total_loss = total_loss + d_loss_coef * d_loss

    # ------------------------ GROUP LOSS -----------------------------
    per_layer_group_loss = jtu.tree_map(
        f=lambda layer: layer.conditioned_output.group_loss,
        tree=output.layers_outputs,
    )

    group_loss = jtu.tree_reduce(
        jnp.add,
        tree=per_layer_group_loss,
        initializer=jnp.zeros(()),
    ).mean()

    loss = (
        total_loss
        + z_loss_coef * z_loss
        + load_balancing_loss_coef * load_balancing_loss
        + d_loss_coef * d_loss
        + group_loss_coef * group_loss
    )

    return loss, MoxEForwardPassOutput(
        model=output,
        ce_loss=ce_loss,
        z_loss=z_loss,
        load_balance_loss=load_balancing_loss,
        d_loss=d_loss,
        group_loss=group_loss,
    )


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
    batch: tuple[jnp.ndarray, ...],
    z_loss_coef: float,
    load_balancing_loss_coef: float,
    d_loss_coef: float = 0.0,
    group_loss_coef: float = 0.0,
):
    """Computes gradients, loss, and updates metrics for a single micro-batch."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    (loss, logits), grads = grad_fn(model, batch)

    # Calculate metrics for this micro-batch
    perplexity = jnp.exp(loss)
    grad_norm = optax.global_norm(grads)  # Calculate gradient norm
    metrics.update(loss=loss, perplexity=perplexity, grad_norm=grad_norm)

    return loss, grads, grad_norm  # Return grad_norm for potential immediate logging


@nnx.jit
def apply_gradients(
    optimizer: nnx.Optimizer,
    grads: nnx.State,  # Gradients are pytrees matching model state structure
):
    """Apply accumulated gradients to the model using the optimizer."""
    optimizer.update(grads)


@nnx.jit
def eval_step(
    model: xLSTMLMModel,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, ...],
):
    """Perform a single evaluation step."""
    loss, logits = loss_fn(model, batch)
    perplexity = jnp.exp(loss)
    # Note: grad_norm is not computed during evaluation
    metrics.update(loss=loss, perplexity=perplexity)


@hydra.main(config_path="./configs", config_name="train_config", version_base="1.1")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    parser = HfArgumentParser(CustomArgs)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
    args = cast(CustomArgs, args)
    # Ensure CustomArgs dataclass in xlstm_jax._trainer.arguments includes a 'warmup_steps' field.
    if not hasattr(args, "warmup_steps"):
        args.warmup_steps = 500  # Default value if not provided
        logger.warning(
            f"args.warmup_steps not found in config, defaulting to {args.warmup_steps}"
        )

    logger.info("Loading tokenizer...")
    # Load teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    config_dict["vocab_size"] = tokenizer.vocab_size
    print(config_dict)
    config = parse_xlstm_config_dict(config_dict)
    config.pad_token_id = tokenizer.pad_token_id

    # Model instance
    dtype_str = cfg["dtype"]
    logger.info(f"Creating xLSTM model with dtype={dtype_str}...")
    dtype = str2dtype(dtype_str)
    rngs = nnx.Rngs(args.seed)
    model = xLSTMLMModel(config, rngs=rngs, dtype=dtype)

    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_dataset = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=config.context_length,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "length"])

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_dataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=config.context_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_dataset.set_format("numpy", columns=["input_ids", "attention_mask", "length"])

    logger.info("Initializing trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="np",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        drop_last=True,  # Important for grad accumulation if dataset size isn't divisible
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Calculate total training steps for the scheduler
    num_train_micro_batches = len(train_loader)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    steps_per_epoch = num_train_micro_batches // args.gradient_accumulation_steps
    if steps_per_epoch == 0:
        logger.warning(
            f"Number of micro-batches ({num_train_micro_batches}) is less than gradient_accumulation_steps ({args.gradient_accumulation_steps}). Effective steps per epoch is 0. Consider reducing accumulation steps or increasing dataset size."
        )
    max_steps = args.num_train_epochs * steps_per_epoch

    # Define the learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=max_steps - args.warmup_steps,
        end_value=args.learning_rate * 0.1,  # Example: decay to 10% of peak lr
    )

    # Note: If using gradient clipping, apply it *before* adamw
    optimizer_def = optax.chain(
        # optax.clip_by_global_norm(args.max_grad_norm), # Uncomment if needed
        optax.adamw(
            learning_rate=lr_schedule,  # Use the scheduler here
            b1=args.adam_beta1,
            b2=args.adam_beta2,
            weight_decay=args.weight_decay,
        ),
    )
    optimizer = nnx.Optimizer(model, optimizer_def)

    # Add grad_norm to metrics
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
        grad_norm=nnx.metrics.Average(
            "grad_norm"
        ),  # Track average grad norm over accumulation steps
    )
    eval_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        perplexity=nnx.metrics.Average("perplexity"),
    )

    history: dict[str, list[dict[str, float]]] = {
        "train_loss": [],
        "train_perplexity": [],
        "train_grad_norm": [],
        "learning_rate": [],
        "eval_loss": [],
        "eval_perplexity": [],
    }

    # checkpoint manager
    ckpt_dir = Path(args.logging_dir).absolute()
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    checkpointer = ocp.PyTreeCheckpointer()
    CKPT_PREFIX = "state"

    # Start training with progress bar being updated and gradient accumulation
    # if needed and descriptive messages
    global_step = 0  # Tracks optimizer steps
    accumulated_grads = None
    grad_structure = None  # To store the structure for zero init

    logger.info("Starting training loop...")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Micro Batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Effective Batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    logger.info(
        f"  Total micro-batches = {num_train_micro_batches * args.num_train_epochs}"
    )
    logger.info(f"  Total optimization steps = {max_steps}")
    logger.info(f"  Warmup steps = {args.warmup_steps}")

    # --- Training Loop ---
    with tqdm(total=max_steps, desc="Optim Steps") as pbar:
        for epoch in range(args.num_train_epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{args.num_train_epochs}")
            metrics.reset()  # Reset train metrics at the start of each epoch

            for micro_step, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1} Micro-steps", leave=False)
            ):
                # Prepare batch
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"])
                _batch = (input_ids, labels)

                # Compute gradients for the micro-batch and update metrics
                loss, grads, grad_norm = compute_grads_and_metrics(
                    model, metrics, _batch
                )

                # Initialize or accumulate gradients
                if accumulated_grads is None:
                    # Get structure from the first computed grads
                    grad_structure = jtu.tree_map(
                        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), grads
                    )
                    accumulated_grads = jtu.tree_map(jnp.zeros_like, grads)

                accumulated_grads = jtu.tree_map(
                    lambda acc, g: acc + g, accumulated_grads, grads
                )

                # Check if it's time for an optimizer step
                is_update_step = (
                    micro_step + 1
                ) % args.gradient_accumulation_steps == 0
                is_last_micro_batch = (micro_step + 1) == num_train_micro_batches

                if is_update_step or is_last_micro_batch:
                    global_step += 1
                    pbar.update(1)

                    # --- Apply Gradients ---
                    # Optional: Scale accumulated gradients
                    scaled_grads = jtu.tree_map(
                        lambda g: g / args.gradient_accumulation_steps,
                        accumulated_grads,
                    )

                    # Apply the gradients
                    apply_gradients(optimizer, scaled_grads)

                    # Reset accumulated gradients
                    accumulated_grads = jtu.tree_map(jnp.zeros_like, accumulated_grads)

                    # --- Logging ---
                    if global_step % args.logging_steps == 0:
                        computed_metrics = metrics.compute()
                        current_lr = lr_schedule(global_step)  # Get current LR
                        for metric, value in computed_metrics.items():
                            history[f"train_{metric}"].append(
                                {
                                    "step": global_step,
                                    "value": value.item(),
                                }
                            )
                        # Log learning rate
                        history["learning_rate"].append(
                            {
                                "step": global_step,
                                "value": current_lr.item(),  # Log the actual LR
                            }
                        )

                        # Update progress bar description
                        log_data = {
                            f"train_{k}": f"{v.item():.4f}"
                            for k, v in computed_metrics.items()
                        }
                        log_data["lr"] = f"{current_lr.item():.2e}"  # Show current LR
                        pbar.set_postfix(log_data)
                        pbar.refresh()

                        # Reset metrics after logging for the next accumulation cycle
                        metrics.reset()

                    # --- Saving ---
                    if global_step % args.save_steps == 0:
                        logger.info(
                            f"Saving checkpoint at optimization step {global_step}..."
                        )
                        state_dir = ckpt_dir / f"{CKPT_PREFIX}-{global_step}"
                        state = nnx.state(model)  # Get model state
                        checkpointer.save(state_dir, state)
                        logger.info(f"Checkpoint saved to {state_dir}")

                    # Break if max_steps reached (e.g., if last micro batch caused an update)
                    if global_step >= max_steps:
                        break

            # Break outer loop if max_steps reached
            if global_step >= max_steps:
                logger.info(f"Reached max_steps ({max_steps}). Finishing training.")
                break

            # --- Evaluation after each epoch ---
            logger.info(f"Starting evaluation after epoch {epoch + 1}...")
            eval_metrics.reset()
            for batch in tqdm(
                eval_loader,
                desc=f"Evaluating Epoch {epoch + 1}",
                leave=False,
            ):
                input_ids = jnp.array(batch["input_ids"])
                labels = jnp.array(batch["labels"])
                _batch = (input_ids, labels)
                eval_step(model=model, metrics=eval_metrics, batch=_batch)

            computed_eval_metrics = eval_metrics.compute()
            for metric, value in computed_eval_metrics.items():
                history[f"eval_{metric}"].append(
                    {
                        "step": global_step,  # Log eval metrics against the last global step of the epoch
                        "value": value.item(),
                    }
                )

            # Update progress bar description with eval metrics
            eval_log_data = {
                f"eval_{k}": f"{v.item():.4f}" for k, v in computed_eval_metrics.items()
            }
            # Add current LR to eval postfix as well for context
            current_lr_eval = lr_schedule(global_step)
            eval_log_data["lr"] = f"{current_lr_eval.item():.2e}"
            pbar.set_postfix(eval_log_data)
            pbar.refresh()
            logger.info(f"Epoch {epoch + 1} Evaluation Results: {eval_log_data}")

    logger.info("Training completed.")

    # --- Final Saving and Upload ---
    logger.info("Saving final artifacts...")
    artifacts_dir = Path(args.output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics history
    with open(artifacts_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {artifacts_dir / 'train_history.json'}")

    # Save model config
    with open(artifacts_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Model config saved to {artifacts_dir / 'config.json'}")

    # Find and copy the best checkpoint based on eval perplexity (or loss if perplexity not available)
    eval_metric_key = (
        "eval_perplexity"
        if "eval_perplexity" in history and history["eval_perplexity"]
        else "eval_loss"
    )
    if history[eval_metric_key]:
        # Sort ascending (lower is better for loss/perplexity)
        sorted_eval_metric = sorted(history[eval_metric_key], key=lambda x: x["value"])
        best_ckpt_info = sorted_eval_metric[0]
        # Ensure step is integer before using in path formatting
        best_step = int(best_ckpt_info["step"])
        metric_value = best_ckpt_info["value"]
        logger.info(
            f"Best checkpoint based on {eval_metric_key}: step {best_step} with value: {metric_value:.4f}"
        )

        best_ckpt_dir_path = ckpt_dir / f"{CKPT_PREFIX}-{best_step}"
        if best_ckpt_dir_path.exists():
            target_ckpt_path = artifacts_dir / CKPT_PREFIX
            logger.info(
                f"Copying best checkpoint from {best_ckpt_dir_path} to {target_ckpt_path}"
            )
            shutil.copytree(best_ckpt_dir_path, target_ckpt_path, dirs_exist_ok=True)
        else:
            logger.warning(
                f"Best checkpoint directory {best_ckpt_dir_path} not found. Saving the last checkpoint instead."
            )
            # Fallback: save the last checkpoint if the best one isn't found (e.g., if saving failed)
            last_ckpt_step = (
                global_step  # Assuming global_step holds the last step number
            )
            last_ckpt_dir_path = ckpt_dir / f"{CKPT_PREFIX}-{last_ckpt_step}"
            if last_ckpt_dir_path.exists():
                target_ckpt_path = artifacts_dir / CKPT_PREFIX
                logger.info(
                    f"Copying last checkpoint from {last_ckpt_dir_path} to {target_ckpt_path}"
                )
                shutil.copytree(
                    last_ckpt_dir_path, target_ckpt_path, dirs_exist_ok=True
                )
            else:
                logger.error("Could not find last checkpoint to save.")

    else:
        logger.warning(
            "No evaluation metrics found to determine the best checkpoint. Saving the final model state."
        )
        # Save the final model state directly if no checkpoints were saved or eval was skipped
        final_state_path = artifacts_dir / CKPT_PREFIX
        final_state = nnx.state(model)
        checkpointer.save(final_state_path, args=ocp.args.StandardSave(final_state))
        logger.info(f"Final model state saved to {final_state_path}")

    # Save tokenizer
    tokenizer.save_pretrained(artifacts_dir)
    logger.info(f"Tokenizer saved to {artifacts_dir}")

    # Push to Hub
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
            commit_message="Training completed",
        )
        logger.info("Push to Hub completed.")


if __name__ == "__main__":
    main()
