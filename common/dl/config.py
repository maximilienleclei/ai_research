from dataclasses import dataclass, field
from typing import Any

from hydra_zen import make_config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from common.config import BaseSubtaskConfig
from common.dl.datamodule.base import BaseDataModule, BaseDataModuleConfig
from common.dl.litmodule.base import BaseLitModule
from common.utils.hydra_zen import generate_config, generate_config_partial


@dataclass
class DeepLearningSubtaskConfig(BaseSubtaskConfig):
    """Configuration for deep learning subtasks.

    Attributes:
        compile: Whether to use torch.compile() for the neural network module.
            Only applies when device is GPU and CUDA compute capability >= 7.
        ckpt_path: Path to checkpoint for resuming training. Use "last" to
            resume from the most recent checkpoint, or None to start fresh.
    """

    compile: bool = False
    ckpt_path: str | None = "last"


@dataclass
class DeepLearningTaskConfig(
    make_config(
        trainer=generate_config_partial(Trainer),
        datamodule=generate_config(
            BaseDataModule,
            config=generate_config(BaseDataModuleConfig),
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
            {"litmodule/nnmodule": "fnn"},
            {"litmodule/scheduler": "constant"},
            {"litmodule/optimizer": "adamw"},
            {"logger": "wandb"},
            "project",
            "task",
            {"task": None},
        ],
    )
