from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from common.ne.popu.base import BasePopu


class BaseEval(ABC):

    @abstractmethod
    def __call__(self, population: "BasePopu") -> torch.Tensor: ...

    @abstractmethod
    def retrieve_num_inputs_outputs(self) -> tuple[int, int]: ...
