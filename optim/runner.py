"""Test."""

from hydra_zen import ZenStore
from runner import BaseTaskRunner

from .store import store_launcher_configs


class OptimTaskRunner(BaseTaskRunner):
    """Optimization ``task`` runner."""

    @classmethod
    def store_configs(cls: type["OptimTaskRunner"], store: ZenStore) -> None:
        """Stores structured configs.

        .. warning::

            Make sure to call this method if you are overriding it.

        Args:
            store: See :meth:`~.BaseTaskRunner.store_configs`.
        """
        super().store_configs(store)
        store_launcher_configs(store)
