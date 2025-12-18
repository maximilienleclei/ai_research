from abc import ABC, abstractmethod
import torch
from common.ne.popu.base import BasePopu

class BaseAlgo(ABC):

    @abstractmethod
    def __call__(self, population: BasePopu, fitness_scores: torch.Tensor) -> None:
        ...
