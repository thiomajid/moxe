from flax import nnx


class SowKeys:
    ENTROPY_LOSS = "entropy_loss"
    PREDICTED_ENTROPY = "predicted_entropy"
    EMPIRICAL_ENTROPY = "empirical_entropy"
    ROUTER_PROBS = "router_probabilities"
    ROUTER_MODULATED_PROBS = "router_modulated_probs"
    GROUP_LOSS = "group_wise_loss"
    Z_LOSS = "z_loss"
    LOAD_BALANCING_LOSS = "load_balancing_loss"
    EXPERTS_ACTIVATION = "experts_activation"
    EXPERTS_TOKEN_LOAD = "tokens_per_expert"


class EntropyLoss(nnx.Intermediate):
    """
    nnx.Variable type used to collect the predicted entropy loss
    in a conditioned MoE gate.
    """

    pass


class PredictedEntropy(nnx.Intermediate):
    pass


class EmpiricalEntropy(nnx.Intermediate):
    pass


class RouterProbs(nnx.Intermediate):
    pass


class RouterModulatedProbs(nnx.Intermediate):
    pass


class GroupLoss(nnx.Intermediate):
    """
    nnx.Variable type used to collect the group loss in a conditioned MoE gate
    to balance the usage of both mLSTM and sLSTM experts.
    """

    pass


class RouterZLoss(nnx.Intermediate):
    """
    nnx.Variable used to collect the z-loss per MoE gate
    """

    pass


class LoadBalancingLoss(nnx.Intermediate):
    """
    nnx.Variable used to collect the load-balancing loss per MoE gate
    """

    pass


class ExpertsActivation(nnx.Intermediate):
    pass


class ExpertsTokenLoad(nnx.Intermediate):
    pass
