"""Deep learning configuration store registration.

This module registers all Hydra configs for the deep learning framework.
The config hierarchy is organized as follows:

Top-level groups:
- trainer: Lightning Trainer configurations
- logger: W&B logger configurations
- datamodule: Data loading configurations (registered by projects)
- litmodule: Lightning module configurations
  - litmodule/nnmodule: Neural network architectures (fnn, mamba, mamba2, rnn, lstm)
  - litmodule/optimizer: Optimizers (adam, adamw, sgd)
  - litmodule/scheduler: LR schedulers (constant, linear_warmup)

To add new configs:
1. Create the config dataclass or partial
2. Call store(..., name="config_name", group="group/subgroup")
3. Override in YAML with: `override /group/subgroup: config_name`
"""

from hydra_zen import ZenStore
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from common.dl.config import DeepLearningTaskConfig
from common.dl.litmodule.store import store_configs as store_litmodule_configs
from common.store import store_wandb_logger_configs
from common.utils.hydra_zen import generate_config_partial


def store_configs(store: ZenStore) -> None:
    """Register all deep learning framework configs to the Hydra store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(DeepLearningTaskConfig, name="config")
    store_litmodule_configs(store)
    store_basic_trainer_config(store)
    store_wandb_logger_configs(
        store,
        clb=WandbLogger,
    )


def store_basic_trainer_config(store: ZenStore) -> None:
    """Register the base trainer configuration.

    The base trainer config sets:
    - accelerator: from config.device
    - default_root_dir: Lightning checkpoint directory
    - gradient_clip_val: 1.0 for training stability

    Args:
        store: The ZenStore instance to register configs to.
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
