# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from .components.init import small_init_initializer
from .components.util import Identity
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class xLSTMLMModel(nnx.Module):
    """Language model using xLSTM blocks as its backbone.

    This model combines token embeddings with an xLSTM block stack
    and a language modeling head for next token prediction.
    """

    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, rngs: nnx.Rngs, dtype=jnp.float32):
        self.config = config
        self.dtype = dtype

        self.xlstm_block_stack = xLSTMBlockStack(config=config, rngs=rngs, dtype=dtype)
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embedding_dim,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        self.embedding_dropout = (
            nnx.Dropout(rate=config.dropout, rngs=rngs)
            if config.add_embedding_dropout
            else Identity()
        )

        self.lm_head = nnx.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            use_bias=False,
            rngs=rngs,
            param_dtype=dtype,
            dtype=dtype,
        )

        # Create shared embedding parameters if using weight tying
        if config.tie_weights:
            # Create a single shared weight for both embedding and output
            initializer = small_init_initializer(dim=config.embedding_dim)
            init_fn = lambda key, shape: initializer(key, shape, dtype=dtype)  # noqa: E731

            self.shared_weight = nnx.Param(
                jnp.zeros((config.vocab_size, config.embedding_dim), dtype=dtype),
                init_fn=init_fn,
            )
        else:
            self.shared_weight = None

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape [B, S]

        Returns:
            Logits of shape [B, S, vocab_size]
        """

        # Get embedding weights (either shared or dedicated)
        if self.config.tie_weights:
            emb_weight = self.shared_weight
            hidden_states = jnp.take(emb_weight, input_ids, axis=0)
        else:
            hidden_states = self.token_embedding(input_ids)

        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.xlstm_block_stack(hidden_states)

        # Apply language model head
        if self.config.tie_weights:
            # When weights are tied, use a functional linear layer with the shared weights
            logits = jnp.matmul(hidden_states, self.shared_weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def reset_parameters(self, rngs: nnx.Rngs):
        self.xlstm_block_stack.reset_parameters(rngs)

        # small_init_initializer is used to initialize the token embedding
        small_init = small_init_initializer(dim=self.config.embedding_dim)
        self.token_embedding.embedding = nnx.Param(
            jnp.zeros(
                (self.config.vocab_size, self.config.embedding_dim),
                dtype=self.dtype,
            ),
            init_fn=lambda key, shape: small_init(key, shape, dtype=self.dtype),
        )

        if not self.config.tie_weights:
            self.lm_head.kernel = nnx.Param(
                jnp.zeros(
                    (self.config.embedding_dim, self.config.vocab_size),
                    dtype=self.dtype,
                ),
                init_fn=lambda key, shape: small_init(key, shape, dtype=self.dtype),
            )
