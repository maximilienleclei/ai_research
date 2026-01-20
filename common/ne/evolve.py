"""Core neuroevolution loop.

This module implements the generational evolution loop that drives neuroevolution:
1. Mutate networks in the population
2. Evaluate fitness using the evaluator
3. Apply selection pressure using the algorithm

The loop runs for a configurable time budget and logs metrics each generation.
"""

import logging
import time

import torch
from torch import Tensor

from common.ne.algo.base import BaseAlgo
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu
from common.utils.misc import seed_all

log = logging.getLogger(__name__)


def _validate_fitness_scores(fitness_scores: Tensor, generation: int) -> None:
    """Check fitness scores for invalid values (NaN/Inf).

    Parameters
    ----------
    fitness_scores : Tensor
        Fitness scores from evaluation.
    generation : int
        Current generation number for error reporting.

    Raises
    ------
    ValueError
        If fitness scores contain NaN or Inf values.
    """
    if torch.isnan(fitness_scores).any():
        raise ValueError(f"Generation {generation}: fitness scores contain NaN values")
    if torch.isinf(fitness_scores).any():
        raise ValueError(f"Generation {generation}: fitness scores contain Inf values")


def _format_metrics_log(
    generation: int, fitness_scores: Tensor, metrics: dict[str, Tensor]
) -> str:
    """Format metrics for logging output.

    Parameters
    ----------
    generation : int
        Current generation number.
    fitness_scores : Tensor
        Overall fitness scores.
    metrics : dict[str, Tensor]
        Additional metrics from evaluator.

    Returns
    -------
    str
        Formatted log message.
    """
    msg = (
        f"Gen {generation}: "
        f"fitness best={fitness_scores.max():.2f} mean={fitness_scores.mean():.2f}"
    )

    # Add generator fitness if available (adversarial evaluators)
    if "fitness_G" in metrics:
        fitness_G = metrics["fitness_G"]
        msg += f" | fitness_G best={fitness_G.max():.2f} mean={fitness_G.mean():.2f}"

    # Add discriminator fitness if available (adversarial evaluators)
    if "fitness_D" in metrics:
        fitness_D = metrics["fitness_D"]
        msg += f" | fitness_D best={fitness_D.max():.2f} mean={fitness_D.mean():.2f}"

    # Add environment rewards if available
    if "env_rewards" in metrics:
        env_rewards = metrics["env_rewards"]
        msg += f" | env_reward best={env_rewards.max():.1f} mean={env_rewards.mean():.1f}"

    # Add target agent rewards if available (imitation evaluators)
    if "target_env_rewards" in metrics:
        target_rewards = metrics["target_env_rewards"]
        msg += f" | target_reward={target_rewards.mean():.1f}"

    return msg


def evolve(
    algo: BaseAlgo,
    eval: BaseEval,
    popu: BasePopu,
    config: NeuroevolutionSubtaskConfig,
) -> None:
    """Run the neuroevolution loop.

    Executes a generational loop of mutation, evaluation, and selection for
    the configured time budget. Logs fitness and evaluator metrics each generation.

    Parameters
    ----------
    algo : BaseAlgo
        Selection algorithm (e.g., SimpleGA with truncation selection).
    eval : BaseEval
        Fitness evaluator (e.g., GymScoreEval for environment rewards).
    popu : BasePopu
        Population of networks to evolve.
    config : NeuroevolutionSubtaskConfig
        Configuration including seed, num_minutes, and output_dir.

    Notes
    -----
    The loop terminates based on wall-clock time (config.num_minutes), not
    generation count. This ensures consistent compute budgets across different
    network complexities.
    """
    seed_all(config.seed)
    log.info(config.output_dir[config.output_dir.find("results/") :])

    start_time = time.time()
    generation = 0

    while (time.time() - start_time) / 60 < config.num_minutes:
        # Evolution step: mutate -> evaluate -> select
        popu.nets.mutate()
        fitness_scores = eval(popu, generation)

        # Validate fitness scores before selection
        _validate_fitness_scores(fitness_scores, generation)

        algo(popu, fitness_scores)

        # Log metrics using standard interface
        metrics = eval.get_metrics()
        msg = _format_metrics_log(generation, fitness_scores, metrics)
        print(msg)

        generation += 1
