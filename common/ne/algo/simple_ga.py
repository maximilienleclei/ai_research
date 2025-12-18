import torch

from common.ne.algo.base import BaseAlgo
from common.ne.popu.base import BasePopu


class SimpleGA(BaseAlgo):

    def __call__(
        self: "SimpleGA", population: BasePopu, fitness_scores: torch.Tensor
    ) -> None:
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        half = population.nets.config.num_nets // 2
        top_half_indices = sorted_indices[:half]
        indices = torch.cat([top_half_indices, top_half_indices])
        population.nets.resample(indices)
