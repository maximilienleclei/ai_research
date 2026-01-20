"""Simple genetic algorithm with truncation selection.

Implements a minimal evolutionary strategy: keep the top 50% of networks
and duplicate them to refill the population. No crossover is used.
"""

import torch
from torch import Tensor

from common.ne.algo.base import BaseAlgo
from common.ne.popu.base import BasePopu


class SimpleGA(BaseAlgo):
    """Simple genetic algorithm with 50% truncation selection.

    Selection Strategy
    ------------------
    1. Sort networks by fitness (descending)
    2. Keep the top 50% (elites)
    3. Duplicate elites to fill the remaining 50%

    This is equivalent to (μ, λ) selection with μ = λ/2.

    Notes
    -----
    No crossover is performed; diversity comes purely from mutation.
    This simple strategy works well for neuroevolution where mutations
    are the primary source of variation.

    Examples
    --------
    >>> algo = SimpleGA()
    >>> fitness = torch.tensor([0.1, 0.5, 0.3, 0.9])  # 4 networks
    >>> algo(population, fitness)
    >>> # Population now contains: [net3, net1, net3, net1] (top 2 duplicated)
    """

    def __call__(
        self: "SimpleGA", population: BasePopu, fitness_scores: Tensor
    ) -> None:
        """Apply 50% truncation selection.

        Args:
            population: Population to select from.
            fitness_scores: Fitness scores with shape (num_nets,).
        """
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        half = population.nets.config.num_nets // 2
        top_half_indices = sorted_indices[:half]
        # Duplicate top half to fill population
        indices = torch.cat([top_half_indices, top_half_indices])
        population.nets.resample(indices)
