from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from common.ne.popu.nets.base import BaseNets


@dataclass
class BasePopuConfig:
    size: int


class BasePopu(ABC):
    def __init__(self, config: BasePopuConfig, nets: BaseNets):
        self.config = config
        self.nets = nets

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
