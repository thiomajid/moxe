import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from xlstm_jax.components.ln import LayerNorm
from xlstm_jax.mask import apply_padding_mask_with_gradient_stop, create_padding_mask

from ..config import MoxEConfig
from ..output import MoxECausalLMOutput, MoxELayerOutput, MoxEModelOutput
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
        self.pad_token_idx = config.xlstm.pad_token_idx

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

    def __call__(
        self,
        input_ids: jax.Array,
        return_layers_outputs: bool = False,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
        training: bool = False,
    ):
        h_t = self.token_embedding(input_ids)
        padding_mask = create_padding_mask(input_ids, self.pad_token_idx)
        h_t = apply_padding_mask_with_gradient_stop(h_t, padding_mask)

        h_t = jax.lax.cond(
            isinstance(self.embedding_dropout, nnx.Dropout),
            lambda: self.embedding_dropout(h_t, deterministic=not training),
            lambda: self.embedding_dropout(h_t),
        )

        # @nnx.scan(
        #     length=len(self.layers),
        #     in_axes=(nnx.Carry,),
        #     out_axes=(nnx.Carry, 0),
        # )
        # def _layer_scan(carry):
        #     current_h_t, layer_idx = carry
        #     layer_out: MoxELayerOutput = nnx.switch(
        #         layer_idx,
        #         self._layer_branches,
        #         current_h_t,
        #         compute_d_loss,
        #         compute_group_loss,
        #     )

        #     updated_h_t = layer_out.hidden_states
        #     new_carry = (updated_h_t, layer_idx + 1)
        #     return new_carry, layer_out

        # init_carry = (h_t, jnp.array(0, dtype=jnp.int32))
        # (h_t, _), layers_outputs = _layer_scan(init_carry)

        layers_outputs: tuple[MoxELayerOutput, ...] | None = (
            () if return_layers_outputs else None
        )
        for layer in self.layers:
            layer_out = layer(
                h_t,
                compute_d_loss=compute_d_loss,
                compute_group_loss=compute_group_loss,
            )
            h_t = layer_out.hidden_states

            if return_layers_outputs:
                layers_outputs += (layer_out,)

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
        return_layers_outputs: bool = False,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
        training: bool = False,
    ):
        moe_out = self.moe(
            input_ids,
            return_layers_outputs=return_layers_outputs,
            compute_d_loss=compute_d_loss,
            compute_group_loss=compute_group_loss,
            training=training,
        )

        h_t = moe_out.hidden_states
        h_t = self.norm(h_t)
        logits = self.lm_head(h_t)

        return MoxECausalLMOutput(
            logits=logits,
            hidden_states=h_t if output_hidden_states else None,
            layers_outputs=moe_out.layers_outputs,
        )
