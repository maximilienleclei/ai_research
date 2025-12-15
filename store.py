r""":mod:`ai_repo` `Hydra <https://hydra.cc>`_ config storing."""

from collections.abc import Callable
from typing import Any

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger
from utils.hydra_zen import generate_config_partial


def store_wandb_logger_configs(
    store: ZenStore,
    clb: Callable[..., Any],
) -> None:
    """Stores `Hydra <https://hydra.cc>`_ ``logger`` group configs.

    Config names: ``wandb``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
        clb: `W&B <https://wandb.ai/>`_ initialization callable.
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
