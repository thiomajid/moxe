from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class CustomArgs(TrainingArguments):
    tokenizer: str = field(default="HuggingFaceTB/SmolLM2-135M-Instruct")
    train_dataset_url: str = field(default="allenai/c4")
    train_subset: Optional[str] = field(default=None)
    train_split: str = field(default="train")
    train_samples: int = field(default=10_000)

    eval_dataset_url: str = field(default="allenai/c4")
    eval_subset: Optional[str] = field(default=None)
    eval_split: str = field(default="validation")
    eval_samples: int = field(default=5_000)

    trust_remote_code: bool = field(default=True)
    features: list[str] = field(
        default_factory=list,
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
