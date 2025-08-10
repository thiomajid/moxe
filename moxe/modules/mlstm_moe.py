from copy import deepcopy

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from xlstm_jax.xlstm_block_stack import xLSTMBlockStack

from ..config import MoxEConfig
from ..utils.types import get_expert_modules
from .base_xlstm_moe import xLSTMMoELayer


class mLSTMMoELayer(xLSTMMoELayer):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        super().__init__(
            config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        # use only mLSTM blocks as a sequence mixer for this kind of layer
        mixer_config = deepcopy(config.xlstm)
        mixer_config.num_blocks = 2
        mixer_config.slstm_at = []
        _block_map = [0, 0]
        mixer_config._block_map = ",".join(map(str, _block_map))

        self.sequence_mixer = xLSTMBlockStack(
            mixer_config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.experts = get_expert_modules(
            config,
            mesh=mesh,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )
