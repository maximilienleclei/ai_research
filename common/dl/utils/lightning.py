"""Lightning utilities for training orchestration.

This module provides utilities for:
- Trainer instantiation with proper callbacks and logging
- Automatic batch size tuning (finds max batch size that fits in GPU VRAM)
- Automatic num_workers tuning (finds minimum workers to saturate GPU)
- GPU time measurement for worker optimization
"""

import contextlib
import copy
import logging
import math
import os
import sys
import time
from datetime import timedelta
from functools import partial
from typing import Annotated as An
from typing import Any

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BatchSizeFinder, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from omegaconf import OmegaConf
from torch.distributed import ReduceOp

from common.dl.constants import (
    BATCH_SIZE_EFFICIENCY_THRESHOLD,
    MIN_BATCH_SIZE_FOR_THROUGHPUT_CHECK,
    WORKER_SATURATION_BUFFER,
)
from common.dl.datamodule.base import BaseDataModule
from common.dl.litmodule.base import BaseLitModule
from common.utils.beartype import one_of
from common.utils.hydra import get_launcher_config
from common.utils.misc import can_connect_to_internet

log = logging.getLogger(__name__)


def instantiate_trainer(
    trainer_partial: partial[Trainer],
    logger_partial: partial[WandbLogger],
    device: An[str, one_of("cpu", "gpu")],
    output_dir: str,
    save_every_n_minutes: int | None,
) -> Trainer:
    """Instantiate a Lightning Trainer with configured callbacks and logger.

    Sets up:
    - ModelCheckpoint callback for saving best and last checkpoints
    - W&B logger with Hydra config logging
    - Proper device count based on launcher configuration

    Args:
        trainer_partial: Partial Trainer function with base settings.
        logger_partial: Partial WandbLogger function.
        device: Computing device ("cpu" or "gpu").
        output_dir: Directory for saving checkpoints and logs.
        save_every_n_minutes: Interval for time-based checkpointing.
            If None, only saves at validation epochs.

    Returns:
        Fully configured Trainer instance ready for training.
    """
    launcher_config = get_launcher_config()
    # Retrieve the `Trainer` callbacks specified in the config
    callbacks: list[Any] = trainer_partial.keywords["callbacks"] or []
    # Check internet connection
    offline = (
        logger_partial.keywords["offline"] or not can_connect_to_internet()
    )
    if offline:
        os.environ["WANDB_DISABLE_SERVICE"] = "True"
    # Adds a callback to save the state of training (not just the model
    # despite the name) at the end of every validation epoch.
    callbacks.append(
        ModelCheckpoint(
            dirpath=trainer_partial.keywords["default_root_dir"],
            monitor="val/loss",
            save_last=True,
            save_top_k=1,
            train_time_interval=timedelta(minutes=save_every_n_minutes) if save_every_n_minutes else None,
        ),
    )
    # Instantiate the :class:`WandbLogger`.`
    logger = logger_partial(offline=offline)
    # Feed the Hydra (https://hydra.cc) config to W&B (https://wandb.ai).
    logger.experiment.config.update(
        OmegaConf.to_container(
            OmegaConf.load(f"{output_dir}/.hydra/config.yaml"),
            resolve=True,
            throw_on_missing=True,
        ),
    )
    # Instantiate the trainer.
    return trainer_partial(
        devices=(
            launcher_config.gpus_per_node or 1
            if device == "gpu"
            else launcher_config.tasks_per_node
        ),
        logger=logger,
        callbacks=callbacks,
    )


def set_batch_size_and_num_workers(
    trainer: Trainer,
    datamodule: BaseDataModule,
    litmodule: BaseLitModule,
    device: An[str, one_of("cpu", "gpu")],
    output_dir: str,
) -> None:
    """Automatically tune batch size and num_workers for optimal performance.

    This function orchestrates the tuning process:
    1. Finds optimal batch size (if not fixed) via binary search
    2. Measures GPU processing time per batch
    3. Finds minimum num_workers needed to saturate GPU (if not fixed)
    4. Reduces across distributed workers to ensure consistency

    The tuned values are written to datamodule.per_device_batch_size and
    datamodule.per_device_num_workers.

    Args:
        trainer: The Trainer instance (used for distributed reduction).
        datamodule: Data module to configure.
        litmodule: Lightning module for batch size testing.
        device: Computing device.
        output_dir: Directory for temporary tuning outputs.
    """
    if not datamodule.config.fixed_per_device_batch_size:
        proposed_per_device_batch_size = find_good_per_device_batch_size(
            litmodule=litmodule,
            datamodule=datamodule,
            device=device,
            device_ids=trainer.device_ids,
            output_dir=output_dir,
        )
        per_device_batch_size = int(
            trainer.strategy.reduce(
                torch.tensor(proposed_per_device_batch_size),
                reduce_op="min",
            ),
        )
    else:
        per_device_batch_size = datamodule.config.fixed_per_device_batch_size

    # Measure GPU time for this batch size to inform worker search
    gpu_time = measure_gpu_time_per_batch(
        litmodule, datamodule, device, temp_batch_size=per_device_batch_size
    )

    if not datamodule.config.fixed_per_device_num_workers:
        proposed_per_device_num_workers = find_good_per_device_num_workers(
            datamodule=datamodule,
            per_device_batch_size=per_device_batch_size,
            gpu_time_per_batch=gpu_time,
        )
        per_device_num_workers = int(
            trainer.strategy.reduce(
                torch.tensor(proposed_per_device_num_workers),
                reduce_op="max",  # type: ignore [arg-type]
            ),
        )
    else:
        per_device_num_workers = datamodule.config.fixed_per_device_num_workers

    datamodule.per_device_batch_size = per_device_batch_size
    datamodule.per_device_num_workers = per_device_num_workers


def find_good_per_device_batch_size(
    litmodule: BaseLitModule,
    datamodule: BaseDataModule,
    device: str,
    device_ids: list[int],
    output_dir: str,
) -> int:
    """Find optimal per-device batch size via binary search.

    Args:
        litmodule: Lightning module for testing forward passes.
        datamodule: Data module providing training data.
        device: Computing device ("cpu" or "gpu").
        device_ids: List of GPU device IDs available.
        output_dir: Directory for temporary tuning outputs.

    Returns:
        Optimal batch size that maximizes throughput within VRAM limits.

    This functionality makes the following assumptions to balance
    training speed, memory stability, and convergence:

    Compute Efficiency: Generally, a larger batch size is preferred for
    GPU saturation, provided it fits in VRAM.

    System Stability: To accurately measure GPU VRAM capacity without
    confounding factors, the search is performed with ``num_workers=0``.
    This prevents System RAM OOMs (caused by dataloader processes) from
    being misinterpreted as GPU VRAM OOMs.

    Diminishing Returns: A larger batch size is only accepted if it
    provides a tangible throughput increase (>5% samples/sec). If doubling
    the batch size results in linearly slower step times (no gain in
    samples/sec), the smaller batch size is preferred to maintain better
    gradient stochasticity and generalization.

    Multi-GPU: If training on multiple GPUs, it is assumed each GPU
    has roughly the same amount of VRAM.
    """
    launcher_config = get_launcher_config()
    datamodule_copy = copy.deepcopy(datamodule)
    datamodule_copy.prepare_data()
    datamodule_copy.setup("fit")
    # (since the default `datamodule` `batch_size` is 1)
    try:
        dataset_len = len(datamodule_copy.train_dataloader())
    except TypeError:
        dataset_len = 10000  # Fallback if length unknown

    num_computing_devices = launcher_config.nodes * (
        launcher_config.gpus_per_node or 1
        if device == "gpu"
        else launcher_config.tasks_per_node
    )
    per_device_batch_size: int | None
    # Ensures total batch_size is < 1% of the train dataloader size.
    max_per_device_batch_size = dataset_len // (100 * num_computing_devices)
    # If a maximum batch size was specified, use the smaller of the two.
    if datamodule.config.max_per_device_batch_size:
        max_per_device_batch_size = min(
            max_per_device_batch_size,
            datamodule.config.max_per_device_batch_size,
        )
    if device == "cpu":
        # Only running batch size finding on GPU: reaching out of GPU
        # memory errors does not freeze the system wheras reaching out
        # of CPU memory errors does.
        per_device_batch_size = max_per_device_batch_size
    else:
        litmodule_copy = copy.deepcopy(litmodule)
        # Speeds up the batch size search by removing the validation
        # epoch end method, which is independent of the batch size.
        litmodule_copy.on_validation_epoch_end = None  # type: ignore[assignment,method-assign]
        # Speeds up the batch size search by using a reasonable number
        # of workers for the search.
        # CRITICAL CHANGE: We set workers to 0 here to prevent System RAM OOMs
        # from interfering with VRAM discovery.
        datamodule_copy.per_device_num_workers = 0
        batch_size_finder = BatchSizeFinder(
            mode="binsearch",
            batch_arg_name="per_device_batch_size",
            max_trials=int(math.log2(max_per_device_batch_size)) + 1,
        )
        # Stops the `fit` method after the batch size has been found.
        batch_size_finder._early_exit = True  # noqa: SLF001
        trainer = Trainer(
            accelerator=device,
            devices=[device_ids[0]],  # The first available device.
            default_root_dir=output_dir + "/lightning/tuner/",
            callbacks=[batch_size_finder],
            enable_checkpointing=False,  # Disable checkpointing for speed
            logger=False,
        )
        log.info("Finding good `batch_size` parameter...")
        per_device_batch_size = None
        # Prevents the `fit` method from raising a `KeyError`, see:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18114
        with contextlib.suppress(KeyError):
            trainer.fit(model=litmodule_copy, datamodule=datamodule_copy)
        # If any OOM occurs the value is stored in
        # `per_device_batch_size` else in `optimal_batch_size`.
        found_batch_size = max(
            datamodule_copy.per_device_batch_size,
            batch_size_finder.optimal_batch_size or 1,
        )
        found_batch_size = min(
            found_batch_size,
            max_per_device_batch_size,
        )
        # If the largest batch fits but is slower (due to compute saturation),
        # prefer the smaller batch for better convergence properties.
        if found_batch_size > MIN_BATCH_SIZE_FOR_THROUGHPUT_CHECK and device != "cpu":
            log.info(
                f"Max safe batch size found: {found_batch_size}. Verifying throughput..."
            )
            candidates = [found_batch_size // 2, found_batch_size]
            throughputs = []
            for b_size in candidates:
                t_per_batch = measure_gpu_time_per_batch(
                    litmodule_copy,
                    datamodule_copy,
                    device,
                    temp_batch_size=b_size,
                )
                if t_per_batch > 0:
                    throughputs.append(b_size / t_per_batch)  # samples/sec
                else:
                    throughputs.append(0)
            small_throughput = throughputs[0]
            large_throughput = throughputs[1]
            log.info(
                f"Half Batch ({candidates[0]}): {small_throughput:.1f} samp/s"
            )
            log.info(
                f"Max Batch  ({candidates[1]}): {large_throughput:.1f} samp/s"
            )
            # Compare: Is the larger batch sufficiently more efficient?
            if large_throughput < (small_throughput * BATCH_SIZE_EFFICIENCY_THRESHOLD):
                log.info(
                    "Larger batch yielded diminishing returns. Reverting to smaller batch."
                )
                per_device_batch_size = candidates[0]
            else:
                per_device_batch_size = candidates[1]
        else:
            per_device_batch_size = found_batch_size
    if per_device_batch_size == 0:
        per_device_batch_size = 1
    log.info(f"Best `batch_size` parameter: {per_device_batch_size}.")
    return per_device_batch_size


def find_good_per_device_num_workers(
    datamodule: BaseDataModule,
    per_device_batch_size: int,
    gpu_time_per_batch: float,
    max_num_data_passes: int = 50,
) -> int:
    """Find minimum num_workers needed to keep GPU saturated.

    Args:
        datamodule: Data module to configure.
        per_device_batch_size: The batch size to use for testing.
        gpu_time_per_batch: Measured GPU processing time per batch (seconds).
            Used as the target to beat for dataloader speed.
        max_num_data_passes: Maximum batches to iterate per worker config.

    Returns:
        Minimum number of workers that can produce batches faster than
        the GPU can consume them.

    Iterates through increasing values of ``num_workers`` to find the
    minimum number of workers required to saturate the GPU.

    This functionality relies on the following logic:

    Saturation Point:* The goal is not maximum raw dataloader speed,
    but rather to be "just faster" than the GPU. Once the dataloader
    can produce a batch faster than the GPU can consume it (based on
    `gpu_time_per_batch`), the search stops.

    Resource Conservation: By stopping early at the saturation
    point, we avoid spawning unnecessary worker processes, preserving
    System RAM and reducing CPU context-switching overhead.

    Forward Iteration: The search proceeds upwards (0, 2, 4...)
    rather than downwards to catch the "knee" of the performance curve
    efficiently.
    """
    launcher_config = get_launcher_config()
    log.info("Finding good `num_workers` parameter...")
    if launcher_config.cpus_per_task in [None, 1]:
        log.info("Only 1 worker available/provided. Returning 0.")
        return 0
    # Static type checking purposes, already narrowed down to `int`
    # through the `if` statement above.
    assert launcher_config.cpus_per_task  # noqa: S101
    # CHANGE: Search upwards (0, 2, 4...) instead of downwards to find
    # the knee of the curve and avoid unnecessary overhead.
    candidate_workers = [0] + [
        i for i in range(2, launcher_config.cpus_per_task + 1, 2)
    ]
    candidate_workers = sorted(list(set(candidate_workers)))
    best_workers = 0
    best_time_taken = float("inf")
    datamodule_copy = copy.deepcopy(datamodule)
    datamodule_copy.per_device_batch_size = per_device_batch_size
    datamodule_copy.prepare_data()
    datamodule_copy.setup("fit")
    log.info(f"Target GPU Time to beat: {gpu_time_per_batch:.4f}s")
    for num_workers in candidate_workers:
        datamodule_copy.per_device_num_workers = num_workers
        try:
            loader = datamodule_copy.train_dataloader()
            iterator = iter(loader)
            next(iterator)  # Warmup
        except Exception as e:
            log.warning(f"Failed worker config {num_workers}: {e}")
            continue
        start_time = time.time()
        num_batches = 0
        try:
            # CHANGE: Reduced loop complexity to avoid infinite loops
            for _ in range(max_num_data_passes):
                _ = next(iterator)
                num_batches += 1
                # CRITICAL: Sleep to mimic the GPU working
                # This ensures we don't optimize for CPU throughput
                # that the GPU can't consume.
                if gpu_time_per_batch > 0:
                    time.sleep(gpu_time_per_batch)
        except StopIteration:
            pass
        total_time = time.time() - start_time
        avg_batch_time = total_time / num_batches
        log.info(
            f"num_workers: {num_workers}, avg_batch_time: {avg_batch_time:.4f}s"
        )
        # LOGIC: Saturation Check
        # If this config is already faster than the GPU needs (plus buffer),
        # we stop. We found the "Good Enough" point.
        if gpu_time_per_batch > 0 and avg_batch_time <= (
            gpu_time_per_batch * WORKER_SATURATION_BUFFER
        ):
            log.info(
                f"Saturation reached at {num_workers} workers. Stopping search."
            )
            return num_workers
        if total_time < best_time_taken:
            best_time_taken = total_time
            best_workers = num_workers
        else:
            # If the time taken is not decreasing, stop the search.
            log.info("Diminishing returns detected. Stopping.")
            break
    log.info(f"Best `num_workers` parameter: {best_workers}.")
    return best_workers


def measure_gpu_time_per_batch(
    litmodule: BaseLitModule,
    datamodule: BaseDataModule,
    device: str,
    temp_batch_size: int = None,
    num_samples: int = 20,
) -> float:
    """Measure raw GPU processing time per batch.

    Includes data transfer (CPU->GPU) and forward pass.
    Excludes DataLoader overhead by pre-loading a single batch and reusing it.

    Args:
        litmodule: Lightning module containing the model.
        datamodule: Data module for fetching a test batch.
        device: Computing device ("cpu" or "gpu").
        temp_batch_size: Batch size to use for measurement.
            If None, uses datamodule's current batch size.
        num_samples: Number of forward passes to average over.

    Returns:
        Average time in seconds per batch. Returns 0.0 for CPU device.
    """
    if device == "cpu":
        return 0.0

    dm_copy = copy.deepcopy(datamodule)
    if temp_batch_size:
        dm_copy.per_device_batch_size = temp_batch_size

    # Use 0 workers to minimize setup overhead for this quick test
    dm_copy.per_device_num_workers = 0
    try:
        dm_copy.prepare_data()
        dm_copy.setup("fit")
        loader = dm_copy.train_dataloader()
        iterator = iter(loader)
        batch = next(iterator)  # Fetch one batch
    except Exception as e:
        log.warning(f"Could not fetch batch for measurement: {e}")
        return 0.0

    # Prepare Model
    model = litmodule
    if torch.cuda.is_available() and device != "cpu":
        model = model.to("cuda")
        batch = {
            k: v.to("cuda")
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

    model.eval()  # Use eval to avoid BatchNorm stats updates during check

    # Warmup
    with torch.no_grad():
        _ = model.training_step(batch, batch_idx=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    start_time = time.time()
    for i in range(num_samples):
        with torch.no_grad():
            _ = model.training_step(batch, batch_idx=i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    avg_time = (time.time() - start_time) / num_samples
    return avg_time
