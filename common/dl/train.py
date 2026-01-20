from functools import partial

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from common.dl.config import DeepLearningSubtaskConfig
from common.dl.datamodule.base import BaseDataModule
from common.dl.litmodule.base import BaseLitModule
from common.dl.constants import TORCH_COMPILE_MINIMUM_CUDA_VERSION
from common.dl.utils.lightning import (
    instantiate_trainer,
    set_batch_size_and_num_workers,
)
from common.utils.misc import seed_all


def train(
    trainer: partial[Trainer],
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    logger: partial[WandbLogger],
    config: DeepLearningSubtaskConfig,
) -> float:
    """Train a deep learning model with automatic hyperparameter tuning.

    This function orchestrates the full training pipeline:
    1. Seeds all random number generators for reproducibility
    2. Instantiates the Trainer with appropriate callbacks and logger
    3. Automatically tunes batch size and num_workers (if not fixed)
    4. Optionally applies torch.compile() for GPU acceleration
    5. Runs the training loop with checkpoint resumption support
    6. Returns the final validation loss

    Args:
        trainer: Partial function for Trainer instantiation.
        datamodule: Data module providing train/val dataloaders.
        litmodule: Lightning module containing model and training logic.
        logger: Partial function for WandbLogger instantiation.
        config: Subtask configuration with training settings.

    Returns:
        The validation loss after training completes. This value can be
        used for hyperparameter optimization (HPO) to compare runs.

    Note:
        HPO logic is not yet implemented but the return value is designed
        to support it.
    """
    seed_all(config.seed)
    trainer: Trainer = instantiate_trainer(
        trainer_partial=trainer,
        logger_partial=logger,
        device=config.device,
        output_dir=config.output_dir,
        save_every_n_minutes=config.save_every_n_minutes,
    )
    # TODO: Add logic for HPO - use returned loss to guide search
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
    # TODO: Add logic for HPO - this loss is returned for optimization
    return trainer.validate(model=litmodule, datamodule=datamodule)[0][
        "val/loss"
    ]
