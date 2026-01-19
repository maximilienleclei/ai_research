from abc import ABC, abstractmethod
from typing import Any, final

from hydra_zen import ZenStore, zen
from omegaconf import OmegaConf

from common.config import BaseHydraConfig
from common.store import store_configs as store_common_configs
from common.utils.hydra_zen import destructure
from common.utils.runner import (
    get_absolute_project_path,
    get_project_name,
    get_task_name,
)
from common.utils.wandb import login_wandb


class BaseTaskRunner(ABC):
    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def handle_configs(cls: type["BaseTaskRunner"]) -> None:
        store = ZenStore()
        store(cls.hydra_config, name="config", group="hydra")
        store({"project": get_project_name()}, name="project")
        store({"task": get_task_name()}, name="task")
        # Hydra runtime type checking issues with structured configs:
        # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326
        # `destructure` disables Hydra's runtime type checking, which is
        # fine since we use Beartype throughout the codebase.
        store = store(to_config=destructure)
        cls.store_configs(store=store)
        store.add_to_hydra_store(overwrite_ok=True)

    @final
    @classmethod
    def run_task(cls: type["BaseTaskRunner"]) -> None:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver(
            "replace_slash", lambda s: s.replace("/", ".")
        )
        login_wandb()
        cls.handle_configs()
        zen(cls.run_subtask).hydra_main(
            config_path=get_absolute_project_path(),
            config_name="config",
            version_base=None,
        )

    @classmethod
    def store_configs(cls: type["BaseTaskRunner"], store: ZenStore) -> None:
        store_common_configs(store)

    @staticmethod
    @abstractmethod
    def run_subtask(*args: Any, **kwargs: Any) -> Any: ...
