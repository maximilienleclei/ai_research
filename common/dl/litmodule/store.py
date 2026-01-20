"""Lightning module configuration store registration.

This module registers configs for Lightning modules including:
- Neural network modules (delegated to nnmodule/store.py)
- Optimizers (group: litmodule/optimizer)
- LR schedulers (group: litmodule/scheduler)
- Conditional 1D-to-1D modules (delegated to cond1d_target1d/store.py)

Default configs are specified in DeepLearningTaskConfig.defaults:
- litmodule/nnmodule: fnn
- litmodule/scheduler: constant
- litmodule/optimizer: adamw
"""

from hydra_zen import ZenStore
from torch.optim import SGD, Adam, AdamW
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from common.dl.litmodule.nnmodule.store import (
    store_configs as store_nnmodule_configs,
)
from common.dl.litmodule.cond1d_target1d.store import (
    store_configs as store_cond1d_target1d_configs,
)
from common.utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    """Register all Lightning module configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store_nnmodule_configs(store)
    store_cond1d_target1d_configs(store)
    store_basic_optimizer_configs(store)
    store_basic_scheduler_configs(store)


def store_basic_optimizer_configs(store: ZenStore) -> None:
    """Register basic optimizer configurations.

    Available optimizers:
    - adam: Adam optimizer
    - adamw: AdamW optimizer (default)
    - sgd: Stochastic Gradient Descent

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(
        generate_config_partial(Adam),
        name="adam",
        group="litmodule/optimizer",
    )
    store(
        generate_config_partial(AdamW),
        name="adamw",
        group="litmodule/optimizer",
    )
    store(
        generate_config_partial(SGD),
        name="sgd",
        group="litmodule/optimizer",
    )


def store_basic_scheduler_configs(store: ZenStore) -> None:
    """Register basic LR scheduler configurations.

    Available schedulers:
    - constant: Constant learning rate (default)
    - linear_warmup: Constant LR with linear warmup period

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(
        generate_config_partial(get_constant_schedule),
        name="constant",
        group="litmodule/scheduler",
    )
    store(
        generate_config_partial(get_constant_schedule_with_warmup),
        name="linear_warmup",
        group="litmodule/scheduler",
    )
