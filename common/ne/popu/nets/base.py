from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from common.ne.eval.base import BaseEval


@dataclass
class BaseNetsConfig:
    device: str = "${config.device}"
    num_nets: int = "${popu.config.size}"
    eval: BaseEval = "${eval}"

    def __post_init__(self: "BaseNetsConfig") -> None:
        self.num_inputs, self.num_outputs = (
            self.eval.retrieve_num_inputs_outputs()
        )
        delattr(self, "eval")


class BaseNets(ABC):

    def __init__(self: "BaseNets", config: BaseNetsConfig):
        self.config = config

    @abstractmethod
    def mutate(self: "BaseNets") -> None: ...

    @abstractmethod
    def resample(self: "BaseNets", indices) -> None: ...

    def reset(self: "BaseNets") -> None:
        pass

    @abstractmethod
    def __call__(self: "BaseNets", x: torch.Tensor) -> torch.Tensor: ...
