"""Evolution Strategy for neuroevolution.

Soft selection: all networks contribute weighted by fitness rank.
Operates on flat parameter vectors (via Population.get/set_parameters_flat), NOT network structure.
ONLY works with tensor-based networks (feedforward/recurrent), NOT DynamicNetPopulation.
"""

from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.config import StatePersistenceConfig

from .base import optimize


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


def optimize_es(
    population,
    fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    test_fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
    state_config: StatePersistenceConfig | None = None,
) -> dict:
    """Optimize networks using Evolution Strategy.

    Args:
        population: Population wrapper
        fitness_fn: Training fitness evaluator (closure)
        test_fitness_fn: Test fitness evaluator (closure)
        max_time: Maximum optimization time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path: Path to save/load checkpoints
        logger: Optional logger for tracking
        state_config: Optional state persistence config for recurrent networks

    Returns:
        Dict with fitness_history, test_loss_history, final_generation

    Raises:
        TypeError: If population contains DynamicNetPopulation (raised by select_es)
    """
    return optimize(
        population=population,
        fitness_fn=fitness_fn,
        test_fitness_fn=test_fitness_fn,
        selection_fn=select_es,  # Selection logic defined above
        algorithm_name="es",
        max_time=max_time,
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path,
        logger=logger,
        state_config=state_config,
    )
