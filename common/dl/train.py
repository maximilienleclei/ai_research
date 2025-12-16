from functools import partial

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from common.dl.config import DeepLearningSubtaskConfig
from common.dl.datamodule.base import BaseDataModule
from common.dl.litmodule.base import BaseLitModule
from common.dl.utils.lightning import (
    instantiate_trainer,
    set_batch_size_and_num_workers,
)
from common.utils.misc import seed_all

TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7


def train(
    trainer: partial[Trainer],
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    logger: partial[WandbLogger],
    config: DeepLearningSubtaskConfig,
) -> float:
    seed_all(config.seed)
    trainer: Trainer = instantiate_trainer(
        trainer_partial=trainer,
        logger_partial=logger,
        device=config.device,
        output_dir=config.output_dir,
        save_every_n_train_steps=config.save_every_n_train_steps,
    )
    """TODO: Add logic for HPO"""
    set_batch_size_and_num_workers(
        trainer=trainer,
        datamodule=datamodule,
        litmodule=litmodule,
        device=config.device,
        output_dir=config.output_dir,
    )
    if (
        config.compile
        and config.device == "gpu"
        and torch.cuda.get_device_capability()[0]
        >= TORCH_COMPILE_MINIMUM_CUDA_VERSION
    ):
        litmodule.nnmodule = torch.compile(litmodule.nnmodule)
    if trainer.overfit_batches > 0:
        datamodule.val_dataloader = datamodule.train_dataloader
    trainer.fit(
        model=litmodule,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path,
    )
    """TODO: Add logic for HPO"""
    return trainer.validate(model=litmodule, datamodule=datamodule)[0][
        "val/loss"
    ]
