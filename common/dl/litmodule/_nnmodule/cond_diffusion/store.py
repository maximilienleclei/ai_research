from hydra_zen import ZenStore

from common.dl.litmodule._nnmodule.cond_diffusion.dit_1d_1d import DiT1D1D
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(generate_config(DiT1D1D), name="dit1d1d", group="litmodule/nnmodule")
