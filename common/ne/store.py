"""Neuroevolution configuration store registration.

This module registers all Hydra configs for the neuroevolution framework.
The config hierarchy is organized as follows:

Top-level groups:
- algo: Selection algorithms (simple_ga)
- eval: Fitness evaluators (score_gym, score_torchrl, act_pred, adv_gen, human_act_pred)
- popu: Population wrappers (actor, adv_gen)
  - popu/nets: Network architectures (static_feedforward, static_recurrent, dynamic)

To add new configs:
1. Create the config dataclass or class
2. Call store(..., name="config_name", group="group/subgroup")
3. Override in YAML with: `override /group/subgroup: config_name`
"""

from hydra_zen import ZenStore

from common.ne.config import NeuroevolutionTaskConfig
from common.ne.algo.store import store_configs as store_algo_configs
from common.ne.eval.store import store_configs as store_eval_configs
from common.ne.popu.store import store_configs as store_popu_configs


def store_configs(store: ZenStore) -> None:
    """Register all neuroevolution framework configs to the Hydra store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(NeuroevolutionTaskConfig, name="config")
    store_algo_configs(store)
    store_eval_configs(store)
    store_popu_configs(store)
