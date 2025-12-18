from hydra_zen import ZenStore

from common.ne.popu.nets.base import BaseNets
from common.ne.popu.nets.store import store_configs as store_nets_configs
from common.ne.popu.actor import ActorPopu
from common.ne.popu.base import BasePopuConfig
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store_nets_configs(store)
    store(
        generate_config(ActorPopu, config=generate_config(BasePopuConfig)),
        name="actor",
        group="popu",
    )
