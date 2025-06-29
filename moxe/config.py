from dataclasses import asdict, dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from moxe.output import ModulationBias
from moxe.utils.parser import parse_xlstm_config_dict
from xlstm_jax.xlstm_lm_model import xLSTMLMModelConfig


@dataclass(unsafe_hash=True, order=True)
class MoxEConfig:
    # model_type = "MoxE"

    def __init__(
        self,
        num_experts: int = 4,
        top_k_experts: int = 2,
        num_layers: int = 12,
        moe_layer_type: str = "entropy",
        router_type: str = "entropy_conditioned",
        expert_type: str = "mlp",
        modulation_bias: ModulationBias = ModulationBias.STANDARD,
        gate_bias: bool = False,
        gamma: float = 0.3,
        eps: float = 1e-6,
        difficulty_threshold: float = 0.8,
        ffn_dim: int = 512,
        group_wise_loss: str = "kl_div",
        post_layers_norm: bool = False,
        xlstm: Optional[xLSTMLMModelConfig] = None,
        **kwargs,
    ):
        if xlstm is None:
            xlstm = xLSTMLMModelConfig()

        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.num_layers = num_layers
        self.moe_layer_type = moe_layer_type
        self.router_type = router_type
        self.expert_type = expert_type
        self.modulation_bias = modulation_bias
        self.gate_bias = gate_bias
        self.gamma = gamma
        self.eps = eps
        self.difficulty_threshold = difficulty_threshold
        self.ffn_dim = ffn_dim
        self.group_wise_loss = group_wise_loss
        self.post_layers_norm = post_layers_norm

        self.xlstm = xlstm
        """
        xLSTM related configuration options
        """

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict | DictConfig, **kwargs):
        xlstm_config_dict = None

        if isinstance(config_dict, DictConfig):
            config_dict = OmegaConf.to_container(config_dict, resolve=True)
            xlstm_config_dict = config_dict.pop("xlstm")
        else:
            xlstm_config_dict = config_dict.pop("xlstm")

        xlstm_config = parse_xlstm_config_dict(xlstm_config_dict)
        return cls(xlstm=xlstm_config, **config_dict)
