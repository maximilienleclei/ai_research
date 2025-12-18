from hydra_zen import ZenStore
from mambapy.mamba import MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch.nn import LSTM, RNN

from common.dl.litmodule.nnmodule.cond_autoreg.base import BaseCAMConfig
from common.dl.litmodule.nnmodule.cond_autoreg.discriminative import CDAM
from common.dl.litmodule.nnmodule.cond_autoreg.generative import (
    CGAM,
    CGAMConfig,
)
from common.dl.litmodule.nnmodule.feedforward import FNNConfig
from common.utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    nn_store = store(group="litmodule/nnmodule")
    nn_store(
        generate_config(CDAM, config=generate_config(BaseCAMConfig)),
        name="cdam",
    )
    nn_store(
        generate_config(CGAM, config=generate_config(CGAMConfig)), name="cgam"
    )
    mp_store = store(group="litmodule/nnmodule/model_partial")
    mp_store(generate_config_partial(FNNConfig), name="fnn")
    mp_store(generate_config_partial(MambaConfig), name="mamba")
    mp_store(generate_config_partial(Mamba2Config), name="mamba2")
    mp_store(generate_config_partial(RNN), name="rnn")
    mp_store(generate_config_partial(LSTM), name="lstm")
