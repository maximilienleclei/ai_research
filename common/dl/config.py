from dataclasses import dataclass, field
from typing import Any

from datamodule.base import BaseDataModule, BaseDataModuleConfig
from hydra_zen import make_config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from litmodule.base import BaseLitModule
from utils.hydra_zen import generate_config, generate_config_partial

from ..config import BaseSubtaskConfig


@dataclass
class DeepLearningSubtaskConfig(BaseSubtaskConfig):
    compile: bool = False
    save_every_n_train_steps: int | None = 1
    ckpt_path: str | None = "last"


@dataclass
class DeepLearningTaskConfig(
    make_config(
        trainer=generate_config_partial(Trainer),
        datamodule=generate_config(
            BaseDataModule,
            config=BaseDataModuleConfig(),
        ),
        litmodule=generate_config(BaseLitModule),
        logger=generate_config_partial(WandbLogger),
        config=generate_config(DeepLearningSubtaskConfig),
    ),
):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"trainer": "base"},
            {"litmodule/nnmodule": "mlp"},
            {"litmodule/scheduler": "constant"},
            {"litmodule/optimizer": "adamw"},
            {"logger": "wandb"},
            "project",
            "task",
            {"task": None},
        ],
    )
