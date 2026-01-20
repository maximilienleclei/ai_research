"""Neural network module configuration store registration.

This module registers neural network architecture configs under the
`litmodule/nnmodule` group. Available architectures:

Base architectures:
- fnn: Feedforward Neural Network (MLP) - default
- rnn: Vanilla Recurrent Neural Network
- lstm: Long Short-Term Memory network
- mamba: Mamba state-space model
- mamba2: Mamba2 state-space model

Specialized architectures (delegated to submodules):
- cond_autoreg/*: Conditional autoregressive models
- cond_diffusion/*: Conditional diffusion models

To use a custom nnmodule in YAML:
    defaults:
      - override /litmodule/nnmodule: mamba2
"""

from hydra_zen import ZenStore
from mambapy.mamba import Mamba, MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch import Tensor, nn
from torch.nn import LSTM, RNN

from common.dl.litmodule.nnmodule.cond_autoreg.store import (
    store_configs as store_cond_autoreg_configs,
)
from common.dl.litmodule.nnmodule.cond_diffusion.store import (
    store_configs as store_cond_diffusion_configs,
)
from common.dl.litmodule.nnmodule.feedforward import FNN, FNNConfig
from common.dl.litmodule.nnmodule.mamba2 import Mamba2
from common.utils.hydra_zen import generate_config, generate_config_partial


def store_configs(store: ZenStore) -> None:
    """Register neural network module configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store_cond_autoreg_configs(store)
    store_cond_diffusion_configs(store)
    store = store(group="litmodule/nnmodule")
    store(generate_config(FNN, config=generate_config(FNNConfig)), name="fnn")
    store(
        generate_config(Mamba, config=generate_config(MambaConfig)),
        name="mamba",
    )
    store(
        generate_config(Mamba2, config=generate_config(Mamba2Config)),
        name="mamba2",
    )
    store(generate_config(RNN), name="rnn")
    store(generate_config(LSTM), name="lstm")
