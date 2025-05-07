import typing as tp

import jax
from flax import nnx


class MoELayerType:
    mLSTM = "mlstm"
    sLSTM = "slstm"
    ENTROPY = "entropy"
    MoxE = "moxe"

    @staticmethod
    def values():
        return [
            MoELayerType.mLSTM,
            MoELayerType.sLSTM,
            MoELayerType.MoxE,
        ]


class SparsityGateType:
    STANDARD = "standard"
    CONDITIONED_ADDITION = "conditioned_addition"

    @staticmethod
    def values():
        return [
            SparsityGateType.STANDARD,
            SparsityGateType.CONDITIONED_ADDITION,
        ]


class ExpertModule:
    MLP = "mlp"
    mLSTM = "mlstm"
    sLSTM = "slstm"
    xLSTM = "xlstm"

    @staticmethod
    def values():
        return [
            ExpertModule.MLP,
            ExpertModule.mLSTM,
            ExpertModule.sLSTM,
            ExpertModule.xLSTM,
        ]


class GroupWiseLossFn:
    SELF_BALANCE = "self_balance"
    BOUNDED = "bounded"
    KL_DIV = "kl_div"
    JS_DIV = "js_div"

    @staticmethod
    def values():
        return [
            GroupWiseLossFn.SELF_BALANCE,
            GroupWiseLossFn.BOUNDED,
            GroupWiseLossFn.KL_DIV,
            GroupWiseLossFn.JS_DIV,
        ]


def str2modulation_bias(s: str) -> int:
    if s == "standard":
        return ModulationBias.STANDARD
    elif s == "masked":
        return ModulationBias.MASKED
    elif s == "proportional":
        return ModulationBias.PROPORTIONAL
    else:
        raise ValueError(f"Unknown modulation bias: {s}")


class ModulationBias:
    STANDARD = 0
    MASKED = 1
    PROPORTIONAL = 2

    @staticmethod
    def values():
        return [
            ModulationBias.STANDARD,
            ModulationBias.MASKED,
            ModulationBias.PROPORTIONAL,
        ]


class ConditionedGateOutput(nnx.Module):
    unbiased_logits: jax.Array
    conditioned_logits: jax.Array
    probabilities: jax.Array
    bias: jax.Array
    d_t: jax.Array
    z_loss: jax.Array
    load_balancing_loss: jax.Array
    router_entropy: jax.Array
    predicted_entropy: jax.Array
    expert_load: jax.Array
    expert_token_counts: jax.Array
    d_loss: jax.Array
    group_loss: jax.Array

    def __init__(
        self,
        unbiased_logits: jax.Array,
        conditioned_logits: jax.Array,
        probabilities: jax.Array,
        bias: jax.Array,
        d_t: jax.Array,
        z_loss: jax.Array,
        load_balancing_loss: jax.Array,
        router_entropy: jax.Array,
        predicted_entropy: jax.Array,
        expert_load: jax.Array,
        expert_token_counts: jax.Array,
        d_loss: jax.Array,
        group_loss: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.unbiased_logits = unbiased_logits
        self.conditioned_logits = conditioned_logits
        self.probabilities = probabilities
        self.bias = bias
        self.d_t = d_t
        self.z_loss = z_loss
        self.load_balancing_loss = load_balancing_loss
        self.router_entropy = router_entropy
        self.predicted_entropy = predicted_entropy
        self.expert_load = expert_load
        self.expert_token_counts = expert_token_counts
        self.d_loss = d_loss
        self.group_loss = group_loss


# @flax.struct.dataclass
class MoxELayerOutput(nnx.Module):
    """
    This class is used to store the output of the MoE layer.

    Args:
    - **router_logits**: The logits of the router layer of shape `(B*S, num_experts)`.
    - **hidden_states**: The hidden states of shape `(B, S, hidden_dim)`.
    - **conditioned_output**: The output of the conditioned gate layer of type `ConditionedGateOutput`.
    """

    router_logits: jax.Array
    router_probs: jax.Array
    hidden_states: jax.Array
    z_loss: jax.Array
    load_balancing_loss: jax.Array
    expert_load: jax.Array
    expert_token_counts: jax.Array
    conditioned_output: tp.Optional[ConditionedGateOutput] = None

    def __init__(
        self,
        router_logits: jax.Array,
        router_probs: jax.Array,
        hidden_states: jax.Array,
        z_loss: jax.Array,
        load_balancing_loss: jax.Array,
        expert_load: jax.Array,
        expert_token_counts: jax.Array,
        conditioned_output: tp.Optional[ConditionedGateOutput] = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.router_logits = router_logits
        self.router_probs = router_probs
        self.hidden_states = hidden_states
        self.z_loss = z_loss
        self.load_balancing_loss = load_balancing_loss
        self.expert_load = expert_load
        self.expert_token_counts = expert_token_counts
        self.conditioned_output = conditioned_output


class MoxEModelOutput(nnx.Module):
    hidden_states: jax.Array
    layers_outputs: tp.Optional[tuple[MoxELayerOutput]] = None

    def __init__(
        self,
        hidden_states: jax.Array,
        layers_outputs: tp.Optional[tuple[MoxELayerOutput]] = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.hidden_states = hidden_states
        self.layers_outputs = layers_outputs


class MoxECausalLMOutput(nnx.Module):
    logits: jax.Array
    hidden_states: tp.Optional[jax.Array] = None
    layers_outputs: tp.Optional[tuple[MoxELayerOutput]] = None

    def __init__(
        self,
        logits: jax.Array,
        hidden_states: tp.Optional[jax.Array] = None,
        layers_outputs: tp.Optional[tuple[MoxELayerOutput]] = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.logits = logits
        self.hidden_states = hidden_states
        self.layers_outputs = layers_outputs


class MoxEForwardPassOutput(nnx.Module):
    model: MoxECausalLMOutput
    ce_loss: jax.Array
    z_loss: jax.Array
    load_balance_loss: jax.Array
    d_loss: jax.Array
    group_loss: jax.Array

    def __init__(
        self,
        model: MoxECausalLMOutput,
        ce_loss: jax.Array,
        z_loss: jax.Array,
        load_balance_loss: jax.Array,
        d_loss: jax.Array,
        group_loss: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.model = model
        self.ce_loss = ce_loss
        self.z_loss = z_loss
        self.load_balance_loss = load_balance_loss
        self.d_loss = d_loss
        self.group_loss = group_loss
