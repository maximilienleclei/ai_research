"""Selection algorithm configuration store registration.

This module registers selection algorithm configs under the `algo` group.

Available algorithms:
- simple_ga: 50% truncation selection (top half duplicated to fill population)

To add new algorithms:
1. Create class inheriting BaseAlgo in algo/
2. Register here with store(..., name="algo_name", group="algo")
3. Override in YAML: `override /algo: algo_name`
"""

from hydra_zen import ZenStore

from common.ne.algo.simple_ga import SimpleGA
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    """Register selection algorithm configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(generate_config(SimpleGA), name="simple_ga", group="algo")
