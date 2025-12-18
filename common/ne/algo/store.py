from hydra_zen import ZenStore

from common.ne.algo.simple_ga import SimpleGA
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(generate_config(SimpleGA), name="simple_ga", group="algo")
