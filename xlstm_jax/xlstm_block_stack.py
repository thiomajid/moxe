# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from .components.ln import LayerNorm


@dataclass(unsafe_hash=True, order=True)
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


# @nnx.vmap(in_axes=(0, 0, None, None, None), out_axes=0)
def _slstm_blocks_vmap(
    configs: sLSTMBlockConfig,
    rngs: nnx.Rngs,
    mesh: jax.sharding.Mesh,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
):
    blocks = [
        sLSTMBlock(
            config=block_config,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        for block_config in configs
    ]

    return blocks

    # return sLSTMBlock(
    #     config=block_config,
    #     rngs=rngs,
    #     mesh=mesh,
    #     dtype=dtype,
    #     param_dtype=param_dtype,
    # )


@nnx.vmap(in_axes=(None, 0, None, None, None), out_axes=0)
def _mlstm_blocks_vmap(
    block_config: mLSTMBlockConfig,
    rngs: nnx.Rngs,
    mesh: jax.sharding.Mesh,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
):
    return mLSTMBlock(
        config=block_config,
        rngs=rngs,
        mesh=mesh,
        dtype=dtype,
        param_dtype=param_dtype,
    )


def _create_blocks(
    config: xLSTMBlockStackConfig,
    mesh: jax.sharding.Mesh,
    rngs: nnx.Rngs,
    dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
):
    if all(idx == 0 for idx in config.block_map):
        block_config = deepcopy(config.mlstm_block)
        block_config.__post_init__()

        return _mlstm_blocks_vmap(
            block_config,
            rngs.fork(split=len(config.block_map)),
            mesh,
            dtype,
            param_dtype,
        )
    elif all(idx == 1 for idx in config.block_map):
        slstm_configs = []

        for block_idx, block_type_int in enumerate(config.block_map):
            block_config = deepcopy(config.slstm_block)
            if hasattr(block_config, "_block_idx"):
                block_config._block_idx = block_idx
                block_config.__post_init__()

            slstm_configs.append(block_config)

        return _slstm_blocks_vmap(
            slstm_configs,
            rngs,
            # rngs.fork(split=len(config.block_map)),
            mesh,
            dtype,
            param_dtype,
        )

    blocks: list[mLSTMBlock | sLSTMBlock] = []
    for block_idx, block_type_int in enumerate(config.block_map):
        if block_type_int == 0:
            block_config = deepcopy(config.mlstm_block)
            if hasattr(block_config, "_block_idx"):
                block_config._block_idx = block_idx
                block_config.__post_init__()
            blocks.append(
                mLSTMBlock(
                    config=block_config,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
            )

        elif block_type_int == 1:
            block_config = deepcopy(config.slstm_block)
            if hasattr(block_config, "_block_idx"):
                block_config._block_idx = block_idx
                block_config.__post_init__()
            blocks.append(
                sLSTMBlock(
                    config=block_config,
                    rngs=rngs,
                    mesh=mesh,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
            )

        else:
            raise ValueError(f"Invalid block type {block_type_int}")

    return blocks


@nnx.scan(in_axes=(0, nnx.Carry), out_axes=(nnx.Carry, 0))
def _scan_over_blocks(block: mLSTMBlock | sLSTMBlock, carry: jax.Array):
    next_state = block(carry)
    return next_state, next_state


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
        mesh: jax.sharding.Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.has_uniform_blocks = len(config.slstm_at) == 0
        self.num_blocks = config.num_blocks

        self.blocks = _create_blocks(
            config=config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.post_blocks_norm = (
            LayerNorm(
                num_features=config.embedding_dim,
                use_bias=False,
                rngs=rngs,
                mesh=mesh,
                dtype=dtype,
                param_dtype=param_dtype,
            )
            if config.add_post_blocks_norm
            else jax.nn.identity
        )

    def __call__(self, x: jax.Array):
        """Process input through all blocks in sequence (forward pass).

        Args:
            x: Input tensor of shape [B, S, D]

        Returns:
            Processed output tensor of shape [B, S, D] and hidden states per block
        """

        def _mixed_block_scan(carry: jax.Array, block_idx: jax.Array):
            next_state = jax.lax.switch(block_idx, self.blocks, carry)
            return next_state, next_state

        x_t: jax.Array
        h_t: jax.Array

        if self.has_uniform_blocks:
            x_t, h_t = _scan_over_blocks(self.blocks, x)
        else:
            x_t, h_t = jax.lax.scan(
                f=_mixed_block_scan,
                init=x,
                xs=jnp.arange(self.num_blocks),
            )

        x_t = self.post_blocks_norm(x_t)

        return x_t, h_t
