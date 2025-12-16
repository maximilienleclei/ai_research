from hydra_zen import ZenStore

from utils.hydra_zen import generate_config

from .dit_1d_1d import DiT1D1D


def store_configs(store: ZenStore) -> None:
    store(generate_config(DiT1D1D), name="dit", group="litmodule/nnmodule")
