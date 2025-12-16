from hydra_zen import ZenStore
from mambapy.mamba import Mamba, MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch import Tensor, nn
from torch.nn import LSTM, RNN

from common.dl.litmodule._nnmodule.cond_autoreg.store import (
    store_configs as store_cond_autoreg_configs,
)
from common.dl.litmodule._nnmodule.cond_diffusion.store import (
    store_configs as store_cond_diffusion_configs,
)
from common.dl.litmodule._nnmodule.feedforward import FNN, FNNConfig
from common.dl.litmodule._nnmodule.mamba2 import Mamba2
from common.utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    store_cond_autoreg_configs(store)
    store_cond_diffusion_configs(store)
    store(
        generate_config(FNN, config=generate_config(FNNConfig)),
        name="fnn",
        group="litmodule/nnmodule",
    )
    store(
        generate_config(Mamba, config=generate_config(MambaConfig)),
        name="mamba",
        group="litmodule/nnmodule",
    )
    store(
        generate_config(Mamba2, config=generate_config(Mamba2Config)),
        name="mamba2",
        group="litmodule/nnmodule",
    )
    store(
        generate_config(RNN),
        name="rnn",
        group="litmodule/nnmodule/",
    )
    store(
        generate_config(LSTM),
        name="lstm",
        group="litmodule/nnmodule/",
    )
