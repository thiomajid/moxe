import jax.numpy as jnp
from flax import nnx

from xlstm_jax.components.ln import LayerNorm
from xlstm_jax.components.util import Identity

from ..config import MoxEConfig
from ..output import MoxECausalLMOutput, MoxELayerOutput, MoxEModelOutput
from ..utils.types import get_moe_layer


class MoxEModel(nnx.Module):
    def __init__(self, config: MoxEConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.token_embedding = nnx.Embed(
            num_embeddings=config.xlstm.vocab_size,
            features=config.xlstm.embedding_dim,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.embedding_dropout = (
            nnx.Dropout(
                rate=config.xlstm.dropout,
                rngs=rngs,
            )
            if config.xlstm.add_embedding_dropout
            else Identity()
        )

        layer_type = get_moe_layer(config.moe_layer_type)
        self.layers = [
            layer_type(config, rngs=rngs, dtype=dtype) for _ in range(config.num_layers)
        ]

    def __call__(
        self,
        input_ids: jnp.ndarray,
        return_layers_outputs: bool = False,
    ):
        layers_outputs: tuple[MoxELayerOutput, ...] | None = (
            () if return_layers_outputs else None
        )
        h_t = self.token_embedding(input_ids)
        h_t = self.embedding_dropout(h_t)

        for layer in self.layers:
            layer_out = layer(h_t)
            h_t = layer_out.hidden_states

            if return_layers_outputs:
                layers_outputs += (layer_out,)

        return MoxEModelOutput(layers_outputs=layers_outputs, hidden_states=h_t)


class MoxEForCausalLM(nnx.Module):
    def __init__(self, config: MoxEConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.moe = MoxEModel(config, rngs=rngs, dtype=dtype)
        self.norm = (
            LayerNorm(config.xlstm.embedding_dim, rngs=rngs, dtype=dtype)
            if config.post_layers_norm
            else Identity()
        )

        self.lm_head = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        if config.xlstm.tie_weights:
            # weight tying with token embedding and lm_head
            self.lm_head.kernel = self.moe.token_embedding.embedding

    def __call__(
        self,
        input_ids: jnp.ndarray,
        return_layers_outputs: bool = False,
    ):
        moe_out = self.moe(input_ids, return_layers_outputs=return_layers_outputs)
        h_t = moe_out.hidden_states
        h_t = self.norm(h_t)
        logits = self.lm_head(h_t)

        return MoxECausalLMOutput(
            logits=logits,
            hidden_states=h_t,
            layers_outputs=moe_out.layers_outputs,
        )
