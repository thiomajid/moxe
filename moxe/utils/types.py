import typing as tp
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

from moxe.config import MoxEConfig
from moxe.modules.ffn_expert import FeedForwardExpert
from moxe.output import ExpertModule, MoELayerType
from xlstm_jax.blocks.mlstm.block import mLSTMBlock
from xlstm_jax.blocks.slstm.block import sLSTMBlock

PathLike = Path | str


def get_moe_layer(layer_type: str):
    if layer_type == MoELayerType.mLSTM:
        from moxe.modules.mlstm_moe import mLSTMMoELayer

        return mLSTMMoELayer
    elif layer_type == MoELayerType.sLSTM:
        from moxe.modules.slstm_moe import sLSTMMoELayer

        return sLSTMMoELayer

    elif layer_type == MoELayerType.MoxE:
        from moxe.modules.moxe import MoxELayer

        return MoxELayer

    raise ValueError(
        f"Unknown MoE layer type: {layer_type}"
        f" Supported types are: {MoELayerType.values()}"
    )


def get_expert_modules(config: MoxEConfig, rngs: nnx.Rngs, dtype=jnp):
    if config.expert_type == ExpertModule.MLP:
        return [
            FeedForwardExpert(config, rngs=rngs, dtype=dtype)
            for _ in range(config.num_experts)
        ]

    elif config.expert_type == ExpertModule.mLSTM:
        return [
            mLSTMBlock(config.xlstm.mlstm_block, rngs=rngs, dtype=dtype)
            for _ in range(config.num_experts)
        ]

    elif (
        config.expert_type == ExpertModule.sLSTM
        or config.expert_type == ExpertModule.xLSTM
    ):
        # this hack is necessary otherwise mLSTM blocks are created
        # if not, when the sLSTMBlock is used (same initialization logic as mLSTMMoELayer),
        # the model initialization fails with an error saying that _num_blocks is not defined
        __config = deepcopy(config.xlstm)
        __config.num_blocks = 1
        __config.slstm_at = "all"
        _block_map = [1]
        __config._block_map = ",".join(map(str, _block_map))

        if config.expert_type == ExpertModule.sLSTM:
            return [
                sLSTMBlock(__config.slstm_block, rngs=rngs, dtype=dtype)
                for _ in range(config.num_experts)
            ]

        expert_per_group = config.num_experts // 2
        mlstm_experts = [
            mLSTMBlock(config.xlstm.mlstm_block, rngs=rngs, dtype=dtype)
            for _ in range(expert_per_group)
        ]

        # blocks[0] otherwise an additional LayerNorm is added by xLSTMBlockStack
        sltm_experts = [
            sLSTMBlock(__config.slstm_block, rngs=rngs, dtype=dtype)
            for _ in range(expert_per_group)
        ]

        return mlstm_experts + sltm_experts

    raise ValueError(
        f"Unknown expert module type: {config.expert_type}. "
        f"Supported types are: {ExpertModule.values()}"
    )


_EvalModelKind = tp.Literal["moxe", "xlstm", "hub_model"]


@dataclass
class PerplexityEvaluationConfig:
    def __init__(
        self,
        model_type: _EvalModelKind,
        hub_url: str,
        dataset_url: str,
        data_split: str,
        text_column: str,
        batch_size: int,
        max_seq_length: int,
        num_workers: int,
        local_dir: PathLike | None = None,
        data_subset: tp.Optional[str] = None,
        samples: int | tp.Literal["all"] = "all",
        pin_memory: bool = True,
        hub_token: tp.Optional[str] = None,
    ):
        self.model_type = model_type
        self.local_dir = local_dir
        self.hub_url = hub_url
        self.dataset_url = dataset_url
        self.data_split = data_split
        self.data_subset = data_subset
        self.text_column = text_column
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.samples = samples
        self.pin_memory = pin_memory
        self.hub_token = hub_token

    def __repr__(self):
        return f"PerplexityEvaluationConfig({self.__dict__})"

    def __str__(self):
        return self.__repr__()
