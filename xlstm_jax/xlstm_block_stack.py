# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import jax.numpy as jnp
from flax import nnx

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from .components.ln import LayerNorm
from .components.util import Identity


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None
    slstm_block: Optional[sLSTMBlockConfig] = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: Union[List[int], Literal["all"]] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: Optional[str] = None

    @property
    def block_map(self) -> List[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks

        for slstm_position_idx in self.slstm_at:
            assert slstm_position_idx < self.num_blocks, (
                f"Invalid slstm position {slstm_position_idx}"
            )
            block_map[slstm_position_idx] = 1

        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length

            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()


class xLSTMBlockStack(nnx.Module):
    """Stack of xLSTM blocks that can be either mLSTM or sLSTM blocks.

    This class handles the creation, configuration and sequential processing
    of multiple xLSTM blocks.
    """

    config_class = xLSTMBlockStackConfig

    def __init__(
        self,
        config: xLSTMBlockStackConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.dtype = dtype

        self.blocks = self._create_blocks(config=config, rngs=rngs)

        # Create post-blocks normalization layer
        self.post_blocks_norm = (
            LayerNorm(
                num_features=config.embedding_dim,
                use_bias=False,
                rngs=rngs,
                dtype=dtype,
            )
            if config.add_post_blocks_norm
            else Identity()
        )

    def _create_blocks(self, config: xLSTMBlockStackConfig, rngs: nnx.Rngs):
        """Create blocks according to the block map in the configuration."""
        blocks: list[mLSTMBlock | sLSTMBlock] = []

        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                # Clone configuration to avoid modification issues
                block_config = deepcopy(self.config.mlstm_block)
                if hasattr(block_config, "_block_idx"):
                    block_config._block_idx = block_idx
                    block_config.__post_init__()
                blocks.append(
                    mLSTMBlock(
                        config=block_config,
                        rngs=rngs,
                        dtype=self.dtype,
                    )
                )

            elif block_type_int == 1:
                # Clone configuration to avoid modification issues
                block_config = deepcopy(self.config.slstm_block)
                if hasattr(block_config, "_block_idx"):
                    block_config._block_idx = block_idx
                    block_config.__post_init__()
                blocks.append(
                    sLSTMBlock(
                        config=block_config,
                        rngs=rngs,
                        dtype=self.dtype,
                    )
                )

            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return blocks

    # @nnx.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Process input through all blocks in sequence (forward pass).

        Args:
            x: Input tensor of shape [B, S, D]

        Returns:
            Processed output tensor of shape [B, S, D]
        """

        for block in self.blocks:
            x = block(x)

        x = self.post_blocks_norm(x)

        return x

    def reset_parameters(self, rngs: nnx.Rngs) -> None:
        for block in self.blocks:
            block.reset_parameters(rngs)
        if not isinstance(self.post_blocks_norm, Identity):
            self.post_blocks_norm.reset_parameters(rngs)
