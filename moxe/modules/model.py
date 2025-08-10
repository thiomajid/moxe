import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from moxe.utils.types import get_moe_layer
from xlstm_jax.components.ln import LayerNorm

from ..config import MoxEConfig
from ..output import (
    MoELayerType,
    MoxEForCausalLMOutput,
    MoxELayerOutput,
    MoxEModelOutput,
)


class MoxEModel(nnx.Module):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.pad_token_id = config.xlstm.pad_token_id

        self.token_embedding = nnx.Embed(
            num_embeddings=config.xlstm.vocab_size,
            features=config.xlstm.embedding_dim,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        self.embedding_dropout = (
            nnx.Dropout(
                rate=config.xlstm.dropout,
                rngs=rngs,
            )
            if config.xlstm.add_embedding_dropout
            else jax.nn.identity
        )

        layer_type = get_moe_layer(config.moe_layer_type)
        self.layers = [
            layer_type(config, mesh=mesh, rngs=rngs, dtype=dtype)
            for _ in range(config.num_layers)
        ]

        self.num_layers = config.num_layers
        self.moe_layer_type = config.moe_layer_type

    def __call__(
        self,
        input_ids: jax.Array,
        compute_d_loss: bool = False,
        compute_group_loss: bool = False,
    ):
        h_t = self.token_embedding(input_ids)
        h_t = self.embedding_dropout(h_t)

        def _moxe_scan(carry: tuple[jax.Array, bool, bool], layer_idx: jax.Array):
            state, compute_d_loss, compute_group_loss = carry
            output: MoxELayerOutput = jax.lax.switch(
                layer_idx,
                self.layers,
                state,
                compute_d_loss,
                compute_group_loss,
            )

            next_state = output.hidden_states
            new_carry = (next_state, compute_d_loss, compute_group_loss)
            return new_carry, output

        def _standard_layer_scan(carry: jax.Array, layer_idx: jax.Array):
            next_state: jax.Array = jax.lax.switch(layer_idx, self.layers, carry)
            return next_state, None

        layers_outputs = None
        if self.moe_layer_type == MoELayerType.MoxE:
            carry = (h_t, compute_d_loss, compute_group_loss)
            (h_t, _, _), layers_outputs = jax.lax.scan(
                f=_moxe_scan,
                init=carry,
                xs=jnp.arange(self.num_layers),
            )
        else:
            h_t, _ = jax.lax.scan(
                f=_standard_layer_scan,
                init=h_t,
                xs=jnp.arange(self.num_layers),
            )

        return MoxEModelOutput(hidden_states=h_t, layers_output=layers_outputs)


class MoxEForCausalLM(nnx.Module):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.moe = MoxEModel(config, mesh=mesh, rngs=rngs, dtype=dtype)
        self.norm = (
            LayerNorm(
                config.xlstm.embedding_dim,
                use_bias=False,
                mesh=mesh,
                rngs=rngs,
                dtype=dtype,
            )
            if config.post_layers_norm
            else jax.nn.identity
        )

        self.lm_head = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=config.xlstm.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
        )

        if config.xlstm.tie_weights:
            self.lm_head.kernel = self.moe.token_embedding.embedding

    def __call__(
        self,
        input_ids: jax.Array,
        output_hidden_states: bool = False,
        compute_d_loss: bool = False,
        compute_group_loss: bool = False,
    ):
        moe_out = self.moe(
            input_ids,
            compute_d_loss=compute_d_loss,
            compute_group_loss=compute_group_loss,
        )

        h_t = moe_out.hidden_states
        h_t = self.norm(h_t)
        logits = self.lm_head(h_t)

        return MoxEForCausalLMOutput(
            logits=logits,
            hidden_states=h_t if output_hidden_states else None,
            layers_output=moe_out.layers_output,
        )

    def generate(self, input_ids: jax.Array):
        output = self(input_ids)
        return output.logits
