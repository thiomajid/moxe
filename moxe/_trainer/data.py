import hashlib
import logging
import os
from typing import Literal, Optional, Union

import grain.python as grain
from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from ..config import MoxEConfig
from .arguments import CustomArgs


def get_dataset(
    hub_url: str,
    subset: Optional[str],
    *,
    features: list[str],
    max_seq_length: int,
    tokenizer: AutoTokenizer,
    split: str,
    num_samples: Union[int, Literal["all"]] = "all",
    token: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = "./.dataset_cache",
    trust_remote_code: bool = False,
):
    # Create a unique cache key based on dataset parameters
    cache_key_parts = [
        hub_url,
        str(subset),
        split,
        str(num_samples),
        str(max_seq_length),
        tokenizer.name_or_path,
    ]
    cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()

    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    raw_cache_path = os.path.join(cache_dir, f"raw_{cache_key}")
    tokenized_cache_path = os.path.join(cache_dir, f"tokenized_{cache_key}")

    # Try to load tokenized data from cache
    if use_cache and os.path.exists(tokenized_cache_path):
        try:
            print(f"Loading cached tokenized dataset from {tokenized_cache_path}")
            return load_from_disk(tokenized_cache_path)
        except Exception as e:
            print(f"Failed to load tokenized cache: {e}. Re-processing data.")

    # Try to load raw data from cache
    raw_data = None
    if use_cache and os.path.exists(raw_cache_path):
        try:
            print(f"Loading cached raw dataset from {raw_cache_path}")
            raw_data = load_from_disk(raw_cache_path)
        except Exception as e:
            print(f"Failed to load raw cache: {e}. Re-downloading data.")

    # Download data if not cached
    if raw_data is None:
        data_stream: Optional[IterableDataset] = None

        if subset is not None:
            data_stream = load_dataset(
                hub_url,
                subset,
                split=split,
                streaming=True if num_samples != "all" else False,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            data_stream = load_dataset(
                hub_url,
                split=split,
                streaming=True if num_samples != "all" else False,
                token=token,
                trust_remote_code=trust_remote_code,
            )

        data_points = []

        for data_point in tqdm(data_stream, desc=f"Loading the {split} data"):
            data_points.append(data_point)
            if num_samples != "all" and len(data_points) >= num_samples:
                break

        raw_data = HfDataset.from_list(data_points)

        # Cache the raw data
        if use_cache:
            try:
                print(f"Caching raw dataset to {raw_cache_path}")
                raw_data.save_to_disk(raw_cache_path)
            except Exception as e:
                print(f"Failed to cache raw data: {e}")

    def tokenize_text(element):
        encodings = tokenizer(
            element[features[0]],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_length=True,
            return_tensors="pt",
        )
        return encodings

    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing the {split} data",
    )

    # Cache the tokenized data
    if use_cache:
        try:
            print(f"Caching tokenized dataset to {tokenized_cache_path}")
            tokenized_data.save_to_disk(tokenized_cache_path)
        except Exception as e:
            print(f"Failed to cache tokenized data: {e}")

    return tokenized_data


class HubDataSource(grain.RandomAccessDataSource):
    def __init__(self, dataset: HfDataset) -> None:
        self._dataset = dataset

    def __getitem__(self, record_key):
        return self._dataset[record_key]

    def __len__(self) -> int:
        return len(self._dataset)


class DataCollatatorTransform(grain.MapTransform):
    """
    Applies a collator to a dataset element and converts the specified columns to **JAX** arrays.
    This transform uses a Hugging Face **DataCollatorForLanguageModeling** to process a dataset element,
    then converts the specified columns to **JAX** arrays, removing any other columns.

    Attributes:
        collator: A Hugging Face DataCollatorForLanguageModeling instance.
        target_columns: A list of strings representing the columns to keep and convert to JAX arrays.
    """

    def __init__(
        self,
        collator: DataCollatorForLanguageModeling,
        target_columns: list[str],
    ):
        super().__init__()

        self.collator = collator
        self.target_columns = target_columns

    def map(self, element):
        # if not isinstance(element, list):
        #     element = [element]

        # batch: dict = self.collator.numpy_call(element)
        # result = {}
        # for key in self.target_columns:
        #     if key in batch:
        #         result[key] = jnp.array(batch[key])

        # return result
        return self.collator([element])


def create_dataloaders(
    logger: logging.Logger,
    args: CustomArgs,
    tokenizer: AutoTokenizer,
    config: MoxEConfig,
):
    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_data = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=config.xlstm.context_length,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_data.set_format("numpy", columns=["input_ids", "attention_mask", "length"])
    train_source = HubDataSource(train_data)

    train_sampler = grain.IndexSampler(
        len(train_source),
        shuffle=True,
        seed=args.seed,
        shard_options=grain.NoSharding(),
        num_epochs=int(args.num_train_epochs),
    )

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            grain.Batch(args.per_device_train_batch_size, drop_remainder=True),
            DataCollatatorTransform(
                target_columns=["input_ids", "labels", "attention_mask"],
                collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    return_tensors="np",
                ),
            ),
        ],
    )

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_data = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=config.xlstm.context_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_data.set_format("numpy", columns=["input_ids", "attention_mask", "length"])
    eval_source = HubDataSource(eval_data)

    eval_sampler = grain.IndexSampler(
        len(eval_source),
        shuffle=False,
        seed=args.seed,
        shard_options=grain.NoSharding(),
        num_epochs=int(args.num_train_epochs),
    )

    eval_loader = grain.DataLoader(
        data_source=eval_source,
        sampler=eval_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            grain.Batch(args.per_device_eval_batch_size),
            DataCollatatorTransform(
                target_columns=["input_ids", "labels", "attention_mask"],
                collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                    return_tensors="np",
                ),
            ),
        ],
    )

    return train_loader, eval_loader
