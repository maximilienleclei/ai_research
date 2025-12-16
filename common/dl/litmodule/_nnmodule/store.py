from hydra_zen import ZenStore
from utils.hydra_zen import generate_config, generate_config_partial

from common.dl.litmodule._nnmodule.mlp import MLP, MLPConfig

from .autoregression.store import store_configs as store_autoregression_configs
from .diffusion.store import store_configs as store_diffusion_configs


def store_configs(store: ZenStore) -> None:
    """Stores ``hydra`` :class:`torch.nn.Module` configs.

    Ref: `hydra <https://hydra.cc>`_

    Args:
        store: See :paramref:`~.BaseTaskRunner.store_configs.store`.
    """
    store_diffusion_configs(store)
    store_autoregression_configs(store)
    store(
        generate_config(MLP, config=MLPConfig()),
        name="mlp",
        group="litmodule/nnmodule",
    )
