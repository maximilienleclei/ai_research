"""Population wrapper configuration store registration.

This module registers population configs under the `popu` group.

Available populations:
- actor: Standard action-taking population (discrete or continuous actions)
- adv_gen: Dual-output population for adversarial generation (actions + discrimination)

Network architectures are registered under `popu/nets` (see nets/store.py).

To add new populations:
1. Create class inheriting BasePopu in popu/
2. Register here with store(..., name="popu_name", group="popu")
3. Override in YAML: `override /popu: popu_name`
"""

from hydra_zen import ZenStore

from common.ne.popu.actor import ActorPopu, ActorPopuConfig
from common.ne.popu.adv_gen import AdvGenPopu, AdvGenPopuConfig
from common.ne.popu.nets.store import store_configs as store_nets_configs
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    """Register population configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
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
