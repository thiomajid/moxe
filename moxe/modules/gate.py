import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from moxe.config import MoxEConfig
from moxe.output import ConditionedGateOutput


def _standard_modulation(gamma: float, d_t: jax.Array, _):
    return gamma * d_t


def _proportional_modulation(gamma: float, d_t: jax.Array, difficulty_threshold: float):
    magnitude = d_t - difficulty_threshold
    difficulty = jax.nn.relu(magnitude)
    return gamma * difficulty


def _masked_modulation(gamma: float, d_t: jax.Array, difficulty_threshold: float):
    mask = d_t > difficulty_threshold
    return gamma * d_t * mask.astype(d_t.dtype)


class BiasConditionedGate(nnx.Module):
    def __init__(self, config: MoxEConfig, *, rngs: nnx.Rngs, dtype=jnp.float32):
        self.gamma = config.gamma
        self.num_experts = config.num_experts
        self.experts_per_group = self.num_experts // 2
        self.top_k = config.top_k_experts
        self.difficulty_threshold = config.difficulty_threshold

        self.entropy_predictor = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=1,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.router = nnx.Linear(
            in_features=config.xlstm.embedding_dim,
            out_features=self.num_experts,
            use_bias=config.gate_bias,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.modulation_bias_kind = config.modulation_bias

        self._modulation_fns = [
            _standard_modulation,
            _proportional_modulation,
            _masked_modulation,
        ]

    def __call__(self, h_t: jnp.ndarray):
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
        sigma: jnp.ndarray = lax.switch(
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
        router_raw_logits = self.router(flat_h_t)  # [B*S, num_experts]
        flat_bias = bias.reshape(-1, self.num_experts)  # [B*S, num_experts]
        adjusted_logits = router_raw_logits + flat_bias
        router_probs = jax.nn.softmax(adjusted_logits, axis=-1)  # [B*S, num_experts]

        return ConditionedGateOutput(
            d_t=d_t,
            probabilities=router_probs,
            unbiased_logits=router_raw_logits,
            bias=bias,
        )
