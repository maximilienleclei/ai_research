"""Neuroevolution task runner following common base patterns."""

from functools import partial
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from common.ne.config import NESubtaskConfig
from common.ne.pop.population import Population
from common.ne.store import store_configs as store_ne_configs
from common.ne.train import train
from common.runner import BaseTaskRunner


class NETaskRunner(BaseTaskRunner):
    """Task runner for neuroevolution experiments.

    Extends BaseTaskRunner to handle NE-specific configuration and execution.
    """

    @classmethod
    def store_configs(
        cls: type["NETaskRunner"],
        store: ZenStore,
    ) -> None:
        """Store all NE configurations in Hydra-zen store.

        Args:
            store: Hydra-zen configuration store
        """
        super().store_configs(store)
        store_ne_configs(store)

    @classmethod
    def run_subtask(
        cls: type["NETaskRunner"],
        population: Population,
        train_data: Any,
        test_data: Any,
        logger: partial[WandbLogger],
        config: NESubtaskConfig,
    ) -> Any:
        """Run neuroevolution training subtask.

        Args:
            population: Population wrapper containing networks
            train_data: Training data
            test_data: Test data
            logger: W&B logger partial
            config: NE subtask configuration

        Returns:
            Training results dict
        """
        return train(population, train_data, test_data, logger, config)
