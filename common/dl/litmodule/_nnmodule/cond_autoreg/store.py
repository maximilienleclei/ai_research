from hydra_zen import ZenStore
from mambapy.mamba import MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch.nn import LSTM, RNN

from common.dl.litmodule._nnmodule.cond_autoreg.base import BaseCAMConfig
from common.dl.litmodule._nnmodule.cond_autoreg.discriminative import CDAM
from common.dl.litmodule._nnmodule.cond_autoreg.generative import (
    CGAM,
    CGAMConfig,
)
from common.dl.litmodule._nnmodule.feedforward import FNNConfig
from common.utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(CDAM, config=generate_config(BaseCAMConfig)),
        name="cdam",
        group="litmodule/nnmodule",
    )
    store(
        generate_config(CGAM, config=generate_config(CGAMConfig)),
        name="cgam",
        group="litmodule/nnmodule",
    )
    store(
        generate_config_partial(FNNConfig),
        name="fnn",
        group="litmodule/nnmodule/model_partial",
    )
    store(
        generate_config_partial(MambaConfig),
        name="mamba",
        group="litmodule/nnmodule/model_partial",
    )
    store(
        generate_config_partial(Mamba2Config),
        name="mamba2",
        group="litmodule/nnmodule/model_partial",
    )
    store(
        generate_config_partial(RNN),
        name="rnn",
        group="litmodule/nnmodule/model_partial",
    )
    store(
        generate_config_partial(LSTM),
        name="lstm",
        group="litmodule/nnmodule/model_partial",
    )
