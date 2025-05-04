from copy import deepcopy

import jax.numpy as jnp
from flax import nnx

from moxe.utils.types import get_expert_modules
from xlstm_jax.xlstm_block_stack import xLSTMBlockStack

from ..config import MoxEConfig
from .base_xlstm_moe import xLSTMMoELayer


class sLSTMMoELayer(xLSTMMoELayer):
    def __init__(self, config: MoxEConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        super().__init__(config, rngs=rngs, dtype=dtype)

        self.gate = nnx.Linear(
            config.xlstm.embedding_dim,
            config.num_experts,
            use_bias=config.gate_bias,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        # use only sLSTM blocks as a sequence mixer for this kind of layer
        mixer_config = deepcopy(config.xlstm)
        mixer_config.num_blocks = 2
        mixer_config.slstm_at = "all"
        _block_map = [1, 1]
        mixer_config._block_map = ",".join(map(str, _block_map))
        self.sequence_mixer = xLSTMBlockStack(mixer_config)

        self.experts = get_expert_modules(config)
