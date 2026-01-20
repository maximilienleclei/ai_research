"""Base task runner for Hydra-based projects.

This module provides the base class for running DL and NE tasks with Hydra
configuration management.
"""

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
    """Base class for task runners.

    Provides the infrastructure for:
    - Hydra configuration handling and store registration
    - W&B login
    - Task execution via run_subtask()

    Subclasses must implement run_subtask() with their specific training/evolution logic.
    """

    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def handle_configs(cls: type["BaseTaskRunner"]) -> None:
        """Register all configs to the Hydra store.

        Sets up project name, task name, and calls store_configs() for
        framework-specific config registration.
        """
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
        """Main entry point for running a task.

        Registers OmegaConf resolvers, logs into W&B, handles configs,
        and launches the Hydra main function.
        """
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
        """Register framework-specific configs.

        Override in subclasses to add DL or NE specific configs.

        Args:
            store: The ZenStore instance to register configs to.
        """
        store_common_configs(store)

    @staticmethod
    @abstractmethod
    def run_subtask(*args: Any, **kwargs: Any) -> Any:
        """Execute the subtask with the given configuration.

        This method is called by Hydra with the resolved configuration.
        Must be implemented by subclasses.
        """
        ...
