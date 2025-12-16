"""Simple Genetic Algorithm for neuroevolution.

Hard selection: top 50% survive, bottom 50% replaced by copies of survivors.
Operates on network objects (via Population.select_networks), NOT on parameters.
"""

from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.config import StatePersistenceConfig

from .base import optimize


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


def optimize_ga(
    population,
    fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    test_fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
    state_config: StatePersistenceConfig | None = None,
) -> dict:
    """Optimize networks using Simple Genetic Algorithm.

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
    """
    return optimize(
        population=population,
        fitness_fn=fitness_fn,
        test_fitness_fn=test_fitness_fn,
        selection_fn=select_ga,  # Selection logic defined above
        algorithm_name="ga",
        max_time=max_time,
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path,
        logger=logger,
        state_config=state_config,
    )
