from abc import ABC
from dataclasses import dataclass, field

from common.ne.popu.nets.base import BaseNets, BaseNetsConfig


@dataclass
class StaticNetsConfig(BaseNetsConfig):
    hidden_layer_sizes: list[int] = field(default_factory=list)
    sigma: float = 1e-3
    sigma_sigma: float | None = 1e-2


class BaseStaticNets(BaseNets, ABC):
    def __init__(self: "BaseNets", config: StaticNetsConfig):
        super().__init__(config)
        self.config: StaticNetsConfig
