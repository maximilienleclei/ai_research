"""Deep Learning `Hydra <https://hydra.cc>`_ config store."""

from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from utils.hydra_zen import generate_config_partial


def store_basic_trainer_config(store: ZenStore) -> None:
    """Stores `Hydra <https://hydra.cc>`_ ``trainer`` group configs.

    Config name: ``base``.

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store(
        generate_config_partial(
            Trainer,
            accelerator="${config.device}",
            default_root_dir="${config.output_dir}/lightning/",
            gradient_clip_val=1.0,
        ),
        name="base",
        group="trainer",
    )
