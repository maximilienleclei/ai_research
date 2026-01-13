"""Entry point for training the autoencoder on collected states.

Usage:
    python -m projects.ne_control_score_encoded.train task=train_cartpole
"""

from hydra_zen import ZenStore
from torch.utils.data import DataLoader

from common.dl.runner import DeepLearningTaskRunner
from common.utils.hydra_zen import generate_config, generate_config_partial
from projects.ne_control_score_encoded.datamodule import StateDataModule, StateDataModuleConfig
from projects.ne_control_score_encoded.litmodule import (
    AutoencoderLitModule,
    AutoencoderLitModuleConfig,
)
from projects.ne_control_score_encoded.nnmodule import Autoencoder, AutoencoderConfig


class TaskRunner(DeepLearningTaskRunner):
    """Task runner for autoencoder training on collected states."""

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store=store)
        store(
            generate_config(
                StateDataModule,
                config=generate_config(StateDataModuleConfig),
                dataloader=generate_config_partial(DataLoader),
            ),
            name="state_autoencoder",
            group="datamodule",
        )
        store(
            generate_config(Autoencoder, config=generate_config(AutoencoderConfig)),
            name="autoencoder",
            group="litmodule/nnmodule",
        )
        store(
            generate_config(
                AutoencoderLitModule,
                config=generate_config(AutoencoderLitModuleConfig),
            ),
            name="autoencoder",
            group="litmodule",
        )


TaskRunner.run_task()
