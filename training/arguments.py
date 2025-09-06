import logging
import math
import typing as tp
from dataclasses import dataclass, field


@dataclass(unsafe_hash=True, order=True)
class TrainingArguments:
    tokenizer: str
    dtype: str
    param_dtype: str

    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int

    # optimization
    seed: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    warmup_ratio: float
    max_grad_norm: float

    # logging and checkpoints
    logging_steps: int
    output_dir: str
    logging_dir: str
    run_name: str
    best_metric_key: str
    best_n_to_keep: int

    hub_model_id: str
    hub_token: str
    hub_private_repo: bool
    upload_message: str

    # dataloaders
    train_dataset_url: str = field(default="roneneldan/TinyStories")
    train_subset: tp.Optional[str] = field(default=None)
    train_split: str = field(default="train")
    train_samples: int = field(default=10_000)

    eval_dataset_url: str = field(default="roneneldan/TinyStories")
    eval_subset: tp.Optional[str] = field(default=None)
    eval_split: str = field(default="validation")
    eval_samples: int = field(default=5_000)

    dataloader_drop_last: bool = True
    dataloader_num_workers: int = 4
    worker_buffer_size: int = 2
    text_column: str = field(
        default="text",
        metadata={
            "help": "Columns from the dataset that will be used to create input data for the model"
        },
    )

    use_dataset_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the dataset cache. If False, the dataset will be reloaded every time."
        },
    )

    dataset_cache_dir: str = field(
        default="./.dataset_cache",
        metadata={
            "help": "Directory where the dataset cache will be stored. If not set, the cache will be stored in the current directory."
        },
    )

    # array sharding
    mesh_shape: tuple[int, ...] = field(default_factory=lambda: (1, 8))
    axis_names: tuple[str, ...] = field(default_factory=lambda: ("dp", "tp"))

    def __post_init__(self):
        self.mesh_shape = tuple(self.mesh_shape)
        self.axis_names = tuple(self.axis_names)


class TrainingSteps(tp.NamedTuple):
    train_batches: int
    eval_batches: int
    max_steps: int
    max_optimizer_steps: int
    steps_per_epoch: int
    optimizer_steps_per_epoch: int


def compute_training_steps(
    args: TrainingArguments,
    train_samples: int,
    eval_samples: int,
    logger: logging.Logger,
) -> TrainingSteps:
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    # Calculate number of batches per epoch
    train_batches = train_samples // args.per_device_train_batch_size
    eval_batches = eval_samples // args.per_device_eval_batch_size

    logger.info(f"Batches per epoch - Train: {train_batches}, Eval: {eval_batches}")

    # Each batch is processed as one step
    steps_per_epoch = train_batches

    # Optimizer updates happen every gradient_accumulation_steps batches
    optimizer_steps_per_epoch = math.ceil(
        train_batches // args.gradient_accumulation_steps
    )

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")

    if optimizer_steps_per_epoch == 0:
        logger.warning(
            f"Number of batches per epoch ({train_batches}) is less than gradient_accumulation_steps ({args.gradient_accumulation_steps}). "
            "Effective optimizer steps per epoch is 0. Consider reducing accumulation steps or increasing dataset size."
        )

    max_steps = int(args.num_train_epochs * steps_per_epoch)
    max_optimizer_steps = int(args.num_train_epochs * optimizer_steps_per_epoch)

    return TrainingSteps(
        train_batches=train_batches,
        eval_batches=eval_batches,
        max_steps=max_steps,
        max_optimizer_steps=max_optimizer_steps,
        steps_per_epoch=steps_per_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )
