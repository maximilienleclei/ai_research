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
    store_nnmodule_configs(store)
    store_cond1d_target1d_configs(store)
    store_basic_optimizer_configs(store)
    store_basic_scheduler_configs(store)


def store_basic_optimizer_configs(store: ZenStore) -> None:
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
