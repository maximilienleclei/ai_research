"""Main training entry point for neuroevolution."""

from functools import partial
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers.wandb import WandbLogger

from common.ne.config import NESubtaskConfig
from common.ne.pop.population import Population
from common.utils.misc import seed_all


def train(
    population: Population,
    train_data: Any,
    test_data: Any,
    logger: partial[WandbLogger],
    config: NESubtaskConfig,
) -> dict:
    """Main neuroevolution training entry point.

    Args:
        population: Population wrapper containing networks
        train_data: Training data (format depends on evaluation mode)
        test_data: Test data (format depends on evaluation mode)
        logger: W&B logger partial
        config: NE subtask configuration

    Returns:
        dict with fitness_history, test_loss_history, final_generation
    """
    seed_all(config.seed)

    # Initialize logger if provided
    wb_logger = None
    if logger is not None:
        wb_logger = logger()
        if hasattr(wb_logger, "experiment"):
            wb_logger.experiment  # Force initialization

    # Select optimizer based on config
    if config.optimizer == "ga":
        from common.ne.optim.ga import optimize_ga

        results = optimize_ga(
            population=population,
            fitness_fn=lambda: _create_fitness_fn(population, train_data, config),
            test_fitness_fn=lambda: _create_fitness_fn(population, test_data, config),
            max_time=config.max_time,
            eval_interval=config.eval_interval,
            checkpoint_path=Path(config.checkpoint_path)
            if config.checkpoint_path
            else None,
            logger=wb_logger,
            state_config=config.state_config,
        )

    elif config.optimizer == "es":
        from common.ne.optim.es import optimize_es

        results = optimize_es(
            population=population,
            fitness_fn=lambda: _create_fitness_fn(population, train_data, config),
            test_fitness_fn=lambda: _create_fitness_fn(population, test_data, config),
            max_time=config.max_time,
            eval_interval=config.eval_interval,
            checkpoint_path=Path(config.checkpoint_path)
            if config.checkpoint_path
            else None,
            logger=wb_logger,
            state_config=config.state_config,
        )

    elif config.optimizer == "cmaes":
        from common.ne.optim.cmaes import optimize_cmaes

        results = optimize_cmaes(
            population=population,
            fitness_fn=lambda: _create_fitness_fn(population, train_data, config),
            test_fitness_fn=lambda: _create_fitness_fn(population, test_data, config),
            max_time=config.max_time,
            eval_interval=config.eval_interval,
            checkpoint_path=Path(config.checkpoint_path)
            if config.checkpoint_path
            else None,
            logger=wb_logger,
            state_config=config.state_config,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer}. Must be 'ga', 'es', or 'cmaes'"
        )

    # Close logger
    if wb_logger is not None and hasattr(wb_logger, "finalize"):
        wb_logger.finalize("success")

    return results


def _create_fitness_fn(population: Population, data: Any, config: NESubtaskConfig):
    """Create fitness function closure based on data type.

    This is a placeholder that should be extended based on your evaluation modes.
    For now, it raises NotImplementedError to indicate that specific evaluation
    functions need to be implemented based on the task.

    Args:
        population: Population wrapper
        data: Data for evaluation
        config: Configuration

    Returns:
        Fitness tensor [num_nets]

    Raises:
        NotImplementedError: Specific evaluation logic not yet implemented
    """
    raise NotImplementedError(
        "Specific evaluation logic needs to be implemented based on your task type. "
        "Consider using functions from common.ne.eval.base, common.ne.eval.env, "
        "common.ne.eval.supervised, or common.ne.eval.imitation."
    )
