"""Base class for neuroevolution selection algorithms.

Selection algorithms determine which networks survive and reproduce based on
their fitness scores. They implement the selection pressure that drives evolution.

Common selection strategies include:
- Truncation selection: Keep top K% and duplicate
- Tournament selection: Compete random subsets
- Roulette wheel: Probabilistic selection proportional to fitness
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from common.ne.popu.base import BasePopu


class BaseAlgo(ABC):
    """Abstract base class for selection algorithms.

    Selection algorithms take a population and fitness scores, then resample
    the population to favor high-fitness networks.

    Subclasses
    ----------
    - SimpleGA: 50% truncation selection (keep top half, duplicate)
    """

    @abstractmethod
    def __call__(
        self: "BaseAlgo", population: BasePopu, fitness_scores: Tensor
    ) -> None:
        """Apply selection to the population based on fitness scores.

        This method modifies the population in-place by resampling networks
        according to the selection strategy.

        Args:
            population: Population of networks to select from.
            fitness_scores: Fitness scores with shape (num_nets,). Higher is better.

        Note:
            After this method returns, the population should contain the selected
            networks ready for the next generation's mutation step.
        """
        ...
