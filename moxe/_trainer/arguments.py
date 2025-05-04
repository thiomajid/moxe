from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class CustomArgs(TrainingArguments):
    """
    Arguments pertaining to MoExLSTM training.
    """

    tokenizer: str = field(
        default="HuggingFaceTB/SmolLM2-1.7B",
        metadata={"help": "The tokenizer to use for the model."},
    )

    train_dataset_url: str = field(
        default="roneneldan/TinyStories",
        metadata={"help": "URL to the dataset."},
    )

    eval_dataset_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL to the evaluation dataset."},
    )

    train_split: str = field(
        default="train",
        metadata={"help": "The split to use for training."},
    )

    train_subset: str | None = field(
        default=None,
        metadata={"help": "Subset of the training split to use."},
    )

    train_samples: int = field(
        default=10000,
        metadata={"help": "Number of samples to use for training from the dataset."},
    )

    eval_split: str = field(
        default="validation",
        metadata={"help": "The split to use for evaluation."},
    )

    eval_subset: str | None = field(
        default=None,
        metadata={"help": "Subset of the evaluation split to use."},
    )

    eval_samples: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for evaluation from the dataset."},
    )

    features: list[str] = field(
        default_factory=list,
        metadata={"help": "The features to use from the dataset."},
    )

    use_dataset_cache: bool = field(default=True)
    dataset_cache_dir: str = field(default="./.hf_data_cache")

    monitored_layers: Any = field(
        default="all",
        metadata={"help": "Layers to monitor during training."},
    )

    z_loss_coef: float = field(default=0.001)
    load_balancing_loss_coef: float = field(default=0.01)
    group_loss_coef: float = field(default=0.01)
    d_loss_coef: float = field(default=0.01)
    return_layers_outputs: bool = field(default=True)
    compute_d_loss: bool = field(default=True)
    compute_router_losses: bool = field(default=True)
    compute_group_loss: bool = field(default=True)
    accelerator_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Configuration for the accelerator"},
    )

    trust_remote_code: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        assert self.z_loss_coef >= 0, "Router loss coefficient must be non-negative."
        assert self.load_balancing_loss_coef >= 0, (
            "Load balancing loss coefficient must be non-negative."
        )
        assert self.d_loss_coef >= 0, (
            "Difficulty loss coefficient must be non-negative."
        )
        assert self.group_loss_coef >= 0, "Group loss coefficient must be non-negative."
        # # Initialize accelerator_config if it doesn't exist
        # if self.accelerator_config is None:
        #     self.accelerator_config = {
        #         "gradient_accumulation_steps": self.gradient_accumulation_steps
        #     }

        if self.eval_dataset_url is None:
            self.eval_dataset_url = self.train_dataset_url
