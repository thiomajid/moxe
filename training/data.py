import hashlib
import logging
import os
from typing import Literal, Optional, Union

import grain.python as grain
from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from .arguments import TrainingArguments


def get_dataset(
    hub_url: str,
    subset: Optional[str],
    *,
    text_column: str,
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
            element[text_column],
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


def create_dataloaders(
    logger: logging.Logger,
    args: TrainingArguments,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    target_columns: list[str],
    train_transforms: list[grain.MapTransform],
    eval_transforms: list[grain.MapTransform],
):
    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_data: HfDataset = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        text_column=args.text_column,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    train_data = train_data.select_columns(target_columns)
    train_data.set_format("numpy", columns=target_columns)
    train_source = HubDataSource(train_data)

    train_sampler = grain.IndexSampler(
        len(train_source),
        shuffle=True,
        seed=args.seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=args.dataloader_drop_last),
        num_epochs=1,
    )

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        worker_count=args.dataloader_num_workers,
        worker_buffer_size=args.worker_buffer_size,
        operations=train_transforms,
    )

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_data: HfDataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        text_column=args.text_column,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    logger.info(f"Evaluation dataset loaded with {len(eval_data)} samples")

    eval_data = eval_data.select_columns(target_columns)
    eval_data.set_format("numpy", columns=target_columns)
    eval_source = HubDataSource(eval_data)

    logger.info(f"Evaluation data source created with {len(eval_source)} samples")

    eval_sampler = grain.IndexSampler(
        len(eval_source),
        shuffle=False,
        seed=args.seed,
        shard_options=grain.ShardByJaxProcess(drop_remainder=args.dataloader_drop_last),
        num_epochs=1,
    )

    eval_loader = grain.DataLoader(
        data_source=eval_source,
        sampler=eval_sampler,
        worker_count=args.dataloader_num_workers,
        worker_buffer_size=args.worker_buffer_size,
        operations=eval_transforms,
    )

    return train_loader, eval_loader
