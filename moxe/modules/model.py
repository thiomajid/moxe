import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from xlstm_jax.components.ln import LayerNorm
from xlstm_jax.mask import apply_padding_mask_with_gradient_stop, create_padding_mask

from ..config import MoxEConfig
from ..output import MoELayerType, MoxECausalLMOutput, MoxEModelOutput
from ..utils.types import get_moe_layer


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

        self.num_layers = len(self.layers)
        self.moe_layer_type = config.moe_layer_type

    def __call__(
        self,
        input_ids: jax.Array,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
    ):
        h_t = self.token_embedding(input_ids)
        padding_mask = create_padding_mask(input_ids, self.pad_token_id)
        h_t = apply_padding_mask_with_gradient_stop(h_t, padding_mask)

        h_t = jax.lax.cond(
            isinstance(self.embedding_dropout, nnx.Dropout),
            lambda: self.embedding_dropout(h_t),
            lambda: self.embedding_dropout(h_t),
        )

        # ----------------------------------------------------------------
        # The scan over layers works but bugs with jit compilation
        # ----------------------------------------------------------------

        # @nnx.scan(
        #     in_axes=(None, nnx.Carry, 0),
        #     out_axes=(nnx.Carry, 0),
        #     length=self.num_layers,
        # )
        # def scan_fn(layers, hidden_state, layer_idx: jax.Array):
        #     layer_out = jax.lax.switch(
        #         layer_idx,
        #         layers,
        #         hidden_state,
        #         compute_d_loss,
        #         compute_group_loss,
        #     )

        # updated_state = jax.lax.cond(
        #     self.moe_layer_type == MoELayerType.MoxE,
        #     lambda: layer_out.hidden_states,
        #     lambda: layer_out,
        # )

        #     return updated_state, layer_out

        # h_t, layers_outputs = scan_fn(self.layers, h_t, jnp.arange(self.num_layers))

        layers_outputs = ()
        for layer in self.layers:
            out = layer(h_t, compute_d_loss, compute_group_loss)
            h_t = out.hidden_states if self.moe_layer_type == MoELayerType.MoxE else out
            # h_t = jax.lax.cond(
            #     self.moe_layer_type == MoELayerType.MoxE,
            #     lambda: getattr(out, "hidden_states", out),
            #     lambda: out,
            # )

            layers_outputs = layers_outputs + (out,)

        return MoxEModelOutput(layers_outputs=layers_outputs, hidden_states=h_t)


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
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
    ):
        moe_out = self.moe(
            input_ids,
            compute_d_loss=compute_d_loss,
            compute_group_loss=compute_group_loss,
        )

        h_t = moe_out.hidden_states
        h_t = self.norm(h_t)
        logits = self.lm_head(h_t)

        return MoxECausalLMOutput(
            logits=logits,
            hidden_states=h_t if output_hidden_states else None,
            layers_outputs=moe_out.layers_outputs,
        )
