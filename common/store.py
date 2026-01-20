"""Common configuration store registration.

This module registers shared configs used by both DL and NE frameworks:
- Hydra launchers (group: hydra/launcher)
- W&B logger (group: logger)

These are framework-agnostic utilities used across all projects.
"""

from collections.abc import Callable
from typing import Any

from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from common.utils.hydra_zen import generate_config_partial


def store_configs(store: ZenStore) -> None:
    """Register common framework configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store_launcher_configs(store)


def store_launcher_configs(store: ZenStore) -> None:
    """Register Hydra launcher configs for local and SLURM execution.

    Args:
        store: The ZenStore instance to register configs to.
    """
    # Setting up the launchers is a little bit different from the other
    # configs. Fields get resolved before the ``subtask`` is created.
    args: dict[str, Any] = {  # `generate_config`` does not like dict[str, str]
        "submitit_folder": "${hydra.sweep.dir}/logs/${now:%Y-%m-%d-%H-%M-%S}/",
        "stderr_to_stdout": True,
        "timeout_min": 10080,  # 7 days
    }
    store(LocalQueueConf(**args), group="hydra/launcher", name="local")
    store(SlurmQueueConf(**args), group="hydra/launcher", name="slurm")


def store_wandb_logger_configs(
    store: ZenStore, clb: Callable[..., Any]
) -> None:
    """Register W&B logger configs.

    Args:
        store: The ZenStore instance to register configs to.
        clb: The logger class to configure (WandbLogger or similar).
    """
    dir_key = "save_dir" if clb == WandbLogger else "dir"
    base_args: dict[str, Any] = (
        {  # `generate_config`` does not like dict[str, str]
            "name": "${task}/${hydra:job.override_dirname}",
            dir_key: "${hydra:sweep.dir}/${now:%Y-%m-%d-%H-%M-%S}",
            "project": "${project}",
        }
    )
    store(
        generate_config_partial(clb, **base_args),
        group="logger",
        name="wandb",
    )
