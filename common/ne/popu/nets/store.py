"""Network architecture configuration store registration.

This module registers network configs under the `popu/nets` group.

Available architectures:
- feedforward: Fixed-architecture feedforward networks (weights evolve only)
- recurrent: Fixed-architecture recurrent networks (weights evolve only)
- dynamic: Topology-evolving networks (architecture and weights evolve)

To add new network types:
1. Create class inheriting BaseNets in nets/
2. Implement mutate(), resample(), __call__(), reset()
3. Register here with store(..., name="net_name")
4. Override in YAML: `override /popu/nets: net_name`
"""

from hydra_zen import ZenStore

from common.ne.popu.nets.base import BaseNetsConfig
from common.ne.popu.nets.dynamic.base import DynamicNets
from common.ne.popu.nets.static.base import StaticNetsConfig
from common.ne.popu.nets.static.feedforward import FeedforwardStaticNets
from common.ne.popu.nets.static.recurrent import RecurrentStaticNets
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    """Register network architecture configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
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
