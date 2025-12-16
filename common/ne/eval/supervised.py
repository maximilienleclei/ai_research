"""Supervised learning evaluation orchestration.

Creates fitness evaluators (closures) that capture obs/actions but don't expose them to optimizer.
Provides high-level training interface that orchestrates evaluation + optimization.
"""

from pathlib import Path
from typing import Callable

from jaxtyping import Float, Int
from torch import Tensor


def create_fitness_evaluator(
    population,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, "N"],
) -> Callable[[], Float[Tensor, "num_nets"]]:
    """Create fitness evaluator for supervised learning.

    Returns closure that captures obs/actions internally.
    Optimizer only sees fitness values, never the underlying data.

    Args:
        population: Population wrapper with .nets attribute
        observations: Training observations
        actions: Training actions

    Returns:
        fitness_fn: Callable[[], Tensor] that returns fitness [num_nets]
    """
    from ne.eval.evaluate import evaluate_feedforward, evaluate_recurrent

    nets = population.nets

    # Select evaluation function based on network type
    if hasattr(nets, "forward_batch_sequence"):
        # Recurrent network
        eval_fn = evaluate_recurrent
    elif hasattr(nets, "forward_batch"):
        # Feedforward or dynamic
        eval_fn = evaluate_feedforward
    else:
        raise ValueError(
            f"Unknown network type: {type(nets).__name__}. "
            f"Must implement forward_batch() or forward_batch_sequence()"
        )

    def fitness_fn() -> Float[Tensor, "num_nets"]:
        """Evaluate current population on data.

        Closure captures observations and actions.
        """
        return eval_fn(nets, observations, actions)

    return fitness_fn


def train_supervised(
    population,
    train_data: tuple[Float[Tensor, "N input_size"], Int[Tensor, "N"]],
    test_data: tuple[Float[Tensor, "M input_size"], Int[Tensor, "M"]],
    optimizer: str = "ga",
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
) -> dict:
    """High-level supervised learning training.

    Orchestrates evaluation and optimization with clean separation.
    Optimizer never sees observations/actions, only fitness values.

    Args:
        population: Population wrapper
        train_data: (observations, actions) for training
        test_data: (observations, actions) for testing
        optimizer: Which algorithm to use - "ga", "es", or "cmaes"
        max_time: Maximum training time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path: Optional path to save/load checkpoints
        logger: Optional logger for tracking

    Returns:
        Training results dict with fitness_history, test_loss_history, final_generation

    Raises:
        ValueError: If optimizer is not "ga", "es", or "cmaes"
        TypeError: If using ES/CMA-ES with DynamicNetPopulation
    """
    # Create fitness evaluators (closures that capture data)
    train_obs, train_act = train_data
    test_obs, test_act = test_data

    fitness_fn = create_fitness_evaluator(population, train_obs, train_act)
    test_fitness_fn = create_fitness_evaluator(population, test_obs, test_act)

    # Select and run optimizer
    if optimizer == "ga":
        from ne.optim.ga import optimize_ga

        return optimize_ga(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    elif optimizer == "es":
        from ne.optim.es import optimize_es

        return optimize_es(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    elif optimizer == "cmaes":
        from ne.optim.cmaes import optimize_cmaes

        return optimize_cmaes(
            population=population,
            fitness_fn=fitness_fn,
            test_fitness_fn=test_fitness_fn,
            max_time=max_time,
            eval_interval=eval_interval,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Must be 'ga', 'es', or 'cmaes'"
        )
