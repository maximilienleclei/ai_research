from hydra_zen import ZenStore

from common.ne.popu.nets.base import BaseNetsConfig
from common.ne.popu.nets.dynamic.base import DynamicNets
from common.ne.popu.nets.static.base import StaticNetsConfig
from common.ne.popu.nets.static.feedforward import FeedforwardStaticNets
from common.ne.popu.nets.static.recurrent import RecurrentStaticNets
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store = store(group="popu/nets")
    store(
        generate_config(
            FeedforwardStaticNets, config=generate_config(StaticNetsConfig)
        ),
        name="feedforward",
    )
    store(
        generate_config(
            RecurrentStaticNets, config=generate_config(StaticNetsConfig)
        ),
        name="recurrent",
    )
    store(
        generate_config(DynamicNets, config=generate_config(BaseNetsConfig)),
        name="dynamic",
    )
