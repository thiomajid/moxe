import typing as tp

import jax
from flax import struct


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


def str2modulation_bias(modulation: str) -> int:
    if modulation == "standard":
        return ModulationBias.STANDARD
    elif modulation == "masked":
        return ModulationBias.MASKED
    elif modulation == "proportional":
        return ModulationBias.PROPORTIONAL
    else:
        raise ValueError(f"Unknown modulation bias: {modulation}")


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


@struct.dataclass(unsafe_hash=True, order=True)
class ConditionedGateOutput:
    unbiased_logits: jax.Array
    conditioned_logits: jax.Array
    probabilities: jax.Array
    bias: jax.Array
    d_t: jax.Array
    router_entropy: jax.Array
    predicted_entropy: jax.Array
    d_loss: jax.Array
    group_loss: jax.Array


@struct.dataclass(unsafe_hash=True, order=True)
class BaseMoELayerOutput:
    """
    This class is used to store the output of the MoE layer.

    Metrics are accumulated batch-wise using jax.lax.scan. In other words,
    for a 4-layer MoxE model, the z_loss property will be an array of shape=(4,),
    one scalar per layer.

    Args:
    - **router_logits**: The logits of the router layer of shape `(B*S, num_experts)`.
    - **hidden_states**: The hidden states of shape `(B, S, hidden_dim)`.
    """

    router_logits: jax.Array
    router_probs: jax.Array
    hidden_states: jax.Array
    z_loss: jax.Array
    load_balancing_loss: jax.Array
    expert_load: jax.Array
    expert_token_counts: jax.Array


@struct.dataclass(unsafe_hash=True, order=True)
class MoxELayerOutput(BaseMoELayerOutput):
    """
    This class is used to store the output of a MoxE layer. It inherits properties from
    the dataclass MoELayerOutput.

    Metrics are accumulated batch-wise using jax.lax.scan. In other words,
    for a 4-layer MoxE model, the z_loss property will be an array of shape=(4,),
    one scalar per layer.

    """

    conditioned_output: tp.Optional[ConditionedGateOutput] = None


@struct.dataclass(unsafe_hash=True, order=True)
class MoxEModelOutput:
    hidden_states: jax.Array
    layers_output: tp.Union[BaseMoELayerOutput, MoxELayerOutput]


@struct.dataclass(unsafe_hash=True, order=True)
class MoxEForCausalLMOutput:
    logits: jax.Array
    hidden_states: tp.Optional[jax.Array] = None
    layers_output: tp.Union[BaseMoELayerOutput, MoxELayerOutput] = None


@struct.dataclass(unsafe_hash=True, order=True)
class MoxEForwardPassOutput:
    ce_loss: jax.Array
    z_loss: jax.Array
    load_balancing_loss: jax.Array
    d_loss: jax.Array
    group_loss: jax.Array
    layers_output: tp.Union[BaseMoELayerOutput, MoxELayerOutput]
