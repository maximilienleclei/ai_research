from hydra_zen import ZenStore

from common.ne.popu.actor import ActorPopu, ActorPopuConfig
from common.ne.popu.adv_gen import AdvGenPopu, AdvGenPopuConfig
from common.ne.popu.nets.store import store_configs as store_nets_configs
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store_nets_configs(store)
    store(
        generate_config(ActorPopu, config=generate_config(ActorPopuConfig)),
        name="actor",
        group="popu",
    )
    store(
        generate_config(AdvGenPopu, config=generate_config(AdvGenPopuConfig)),
        name="adv_gen",
        group="popu",
    )
