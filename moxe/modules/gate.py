import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import lax
from jax.sharding import Mesh

from ..config import MoxEConfig
from ..ops import (
    compute_entropy,
    kl_auxiliary_group_loss,
)
from ..output import ConditionedGateOutput, str2modulation_bias


def _standard_modulation(gamma: float, d_t: jax.Array, _):
    return gamma * d_t


def _proportional_modulation(gamma: float, d_t: jax.Array, difficulty_threshold: float):
    magnitude = d_t - difficulty_threshold
    difficulty = jax.nn.relu(magnitude)
    return gamma * difficulty


def _masked_modulation(gamma: float, d_t: jax.Array, difficulty_threshold: float):
    mask = d_t > difficulty_threshold
    return gamma * d_t * mask.astype(d_t.dtype)


def _d_loss_computation(unbiased_logits: jax.Array, d_t: jax.Array):
    router_entropy: jax.Array = compute_entropy(
        jax.nn.softmax(unbiased_logits, axis=-1),
        normalize=True,
    )

    d_t = d_t.reshape(-1)
    loss = optax.losses.squared_error(predictions=d_t, targets=router_entropy).mean()

    return (loss, d_t.mean(), router_entropy.mean())


def _group_wise_loss_computation(
    router_probs: jax.Array,
    batch_size: int,
    seq_len: int,
    num_experts: int,
):
    unflattened_probs = router_probs.reshape(batch_size, seq_len, num_experts)
    experts_per_group = num_experts // 2
    mlstm_usage = unflattened_probs[:, :, :experts_per_group].sum(axis=-1).mean()
    slstm_usage = unflattened_probs[:, :, experts_per_group:].sum(axis=-1).mean()

    return kl_auxiliary_group_loss(pm=mlstm_usage, ps=slstm_usage)


class BiasConditionedGate(nnx.Module):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.gamma = config.gamma
        self.num_experts = config.num_experts
        self.experts_per_group = self.num_experts // 2
        self.top_k = config.top_k_experts
        self.difficulty_threshold = config.difficulty_threshold

        self.entropy_predictor = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=1,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, None),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.router = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=self.num_experts,
            use_bias=config.gate_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

        self.modulation_bias_kind = str2modulation_bias(config.modulation_bias)

        self._modulation_fns = [
            _standard_modulation,
            _proportional_modulation,
            _masked_modulation,
        ]

    def __call__(
        self,
        h_t: jax.Array,
        compute_d_loss: bool = True,
        compute_group_loss: bool = True,
    ):
        """
        Forward pass

        Args:
            hidden_states (torch.Tensor): shape of [B, S, D]

        Returns:
            torch.Tensor: difficulty conditioned gate logits
        """
        batch_size, seq_len, hidden_dim = h_t.shape

        # (B, S, D) -> (B, S, 1)
        d_t = jax.nn.sigmoid(self.entropy_predictor(h_t))

        # threshold_mask = None  # for observability
        sigma: jax.Array = lax.switch(
            self.modulation_bias_kind,
            self._modulation_fns,
            self.gamma,
            d_t,
            self.difficulty_threshold,
        )

        # create the additive bias
        mlstm_bias = sigma  # [B, S, 1]
        slstm_bias = -sigma  # [B, S, 1]

        # Repeat mlstm_bias and slstm_bias along the last dimension
        mlstm_bias = jnp.repeat(mlstm_bias, self.experts_per_group, axis=-1)
        slstm_bias = jnp.repeat(slstm_bias, self.experts_per_group, axis=-1)

        # Concatenate along the last dimension
        # [B, S, total_experts]
        bias = jnp.concatenate([mlstm_bias, slstm_bias], axis=-1)

        # (B, S, D) -> (B*S, D)
        flat_h_t = h_t.reshape(-1, hidden_dim)
        unbiased_logits = self.router(flat_h_t)  # [B*S, num_experts]
        flat_bias = bias.reshape(-1, self.num_experts)  # [B*S, num_experts]
        conditioned_logits = unbiased_logits + flat_bias
        router_probs = jax.nn.softmax(conditioned_logits, axis=-1)

        # ------------------- d_loss computation -----------------------
        d_loss, router_entropy, predicted_entropy = lax.cond(
            compute_d_loss,
            lambda: _d_loss_computation(unbiased_logits=unbiased_logits, d_t=d_t),
            lambda: (jnp.zeros((), dtype=h_t.dtype),) * 3,
        )

        # --------------------- group_wise loss ---------------------------------
        group_loss = lax.cond(
            compute_group_loss,
            lambda: _group_wise_loss_computation(
                router_probs=router_probs,
                batch_size=batch_size,
                seq_len=seq_len,
                num_experts=router_probs.shape[-1],
            ),
            lambda: jnp.zeros((), dtype=h_t.dtype),
        )

        return ConditionedGateOutput(
            unbiased_logits=unbiased_logits,
            conditioned_logits=conditioned_logits,
            probabilities=router_probs,
            bias=bias,
            d_t=d_t,
            router_entropy=router_entropy,
            predicted_entropy=predicted_entropy,
            d_loss=d_loss,
            group_loss=group_loss,
        )


class StandardMoEGate(nnx.Module):
    def __init__(
        self,
        config: MoxEConfig,
        *,
        mesh: Mesh,
        rngs: nnx.Rngs,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    ):
        self.num_experts = config.num_experts

        self.router = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=self.num_experts,
            use_bias=config.gate_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                sharding=(None, "tp"),
                mesh=mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                sharding=("tp",),
                mesh=mesh,
            ),
        )

    def __call__(self, h_t: jax.Array):
        gate_logits = self.router(h_t.reshape(-1, h_t.shape[-1]))
        return gate_logits
