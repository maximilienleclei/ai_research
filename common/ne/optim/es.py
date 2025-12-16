"""Evolution Strategy for neuroevolution.

Soft selection: all networks contribute weighted by fitness rank.
Operates on flat parameter vectors (via Population.get/set_parameters_flat), NOT network structure.
ONLY works with tensor-based networks (feedforward/recurrent), NOT DynamicNetPopulation.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def select_es(population, fitness: Float[Tensor, "num_nets"]) -> None:
    """ES selection: rank-weighted parameter averaging.

    ONLY works with tensor networks - population will raise TypeError for DynamicNetPopulation.

    Args:
        population: Population wrapper
        fitness: Fitness values [num_nets] (lower is better)

    Raises:
        TypeError: If network is DynamicNetPopulation (raised by population.get_parameters_flat())
    """
    num_nets = population.num_nets

    # Get flat parameters [num_nets, num_params]
    params = population.get_parameters_flat()

    # Compute rank-based weights (lower fitness = higher weight)
    ranks = torch.argsort(torch.argsort(fitness))  # Rank 0 = best
    weights = (num_nets - ranks).float()
    weights = weights / weights.sum()  # Normalize to sum to 1

    # Weighted average of parameters [num_nets, num_params] * [num_nets, 1] -> [num_params]
    avg_params = (params * weights.view(-1, 1)).sum(dim=0)

    # Broadcast back to all networks
    new_params = avg_params.unsqueeze(0).expand(num_nets, -1)

    # Set parameters back
    population.set_parameters_flat(new_params)
