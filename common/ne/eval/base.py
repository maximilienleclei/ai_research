from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from common.ne.popu.base import BasePopu


class BaseEval(ABC):

    @abstractmethod
    def __call__(
        self: "BaseEval", population: "BasePopu", generation: int = 0
    ) -> torch.Tensor: ...

    @abstractmethod
    def retrieve_num_inputs_outputs(self: "BaseEval") -> tuple[int, int]: ...

    @abstractmethod
    def retrieve_input_output_specs(self: "BaseEval") -> tuple[Any, Any]: ...
