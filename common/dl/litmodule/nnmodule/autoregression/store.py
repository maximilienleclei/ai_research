from base import BaseCAMConfig
from discriminative import CDAM
from feedforward import FNNConfig
from generative import CGAM, CGAMConfig
from hydra_zen import ZenStore
from mambapy.mamba import MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch.nn import LSTM, RNN

from utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(CDAM, config=BaseCAMConfig()),
        name="cdam",
        group="litmodule/nnmodule",
    )
    store(
        generate_config(CGAM, config=CGAMConfig()),
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
