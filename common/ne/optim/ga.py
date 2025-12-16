"""Simple Genetic Algorithm for neuroevolution.

Hard selection: top 50% survive, bottom 50% replaced by copies of survivors.
Operates on network objects (via Population.select_networks), NOT on parameters.
"""

import torch
from jaxtyping import Float
from torch import Tensor


def select_ga(population, fitness: Float[Tensor, "num_nets"]) -> None:
    """GA selection: top 50% survive and duplicate.

    Works with ALL network types - population handles the network-specific logic.

    Args:
        population: Population wrapper
        fitness: Fitness values [num_nets] (lower is better)
    """
    num_nets = population.num_nets
    num_survivors = num_nets // 2

    # Get survivor indices (top 50% by fitness)
    survivor_indices = torch.argsort(fitness)[:num_survivors]

    # Duplicate survivors to fill population
    indices_with_duplicates = survivor_indices.repeat(2)[:num_nets]

    # Population handles network-specific selection
    population.select_networks(indices_with_duplicates)
