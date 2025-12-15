""":class:`.DeepLearningTaskRunner`."""

from functools import partial
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from common.dl.config import DeepLearningSubtaskConfig, DeepLearningTaskConfig
from common.dl.datamodule import BaseDataModule
from common.dl.litmodule import BaseLitModule
from common.dl.litmodule.store import (store_basic_nnmodule_config,
                                      store_basic_optimizer_configs,
                                      store_basic_scheduler_configs)
from common.dl.store import store_basic_trainer_config
from common.dl.train import train
from common.runner import OptimTaskRunner
from store import store_wandb_logger_configs


class DeepLearningTaskRunner(OptimTaskRunner):
    """Deep Learning ``task`` runner."""

    @classmethod
    def store_configs(
        cls: type["DeepLearningTaskRunner"],
        store: ZenStore,
    ) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store:
                See :paramref:`~.OptimTaskRunner.store_configs.store`.
        """
        super().store_configs(store)
        store_basic_optimizer_configs(store)
        store_basic_scheduler_configs(store)
        store_basic_nnmodule_config(store)
        store_basic_trainer_config(store)
        store_wandb_logger_configs(store, clb=WandbLogger)
        store(DeepLearningTaskConfig, name="config")

    @classmethod
    def run_subtask(
        cls: type["DeepLearningTaskRunner"],
        trainer: partial[Trainer],
        datamodule: BaseDataModule,
        litmodule: BaseLitModule,
        logger: partial[WandbLogger],
        config: DeepLearningSubtaskConfig,
    ) -> Any:  # noqa: ANN401
        """Runs the ``subtask``."""
        return train(trainer, datamodule, litmodule, logger, config)
