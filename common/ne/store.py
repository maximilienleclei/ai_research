from hydra_zen import ZenStore

from common.ne.config import NeuroevolutionTaskConfig
from common.ne.algo.store import store_configs as store_algo_configs
from common.ne.eval.store import store_configs as store_eval_configs
from common.ne.popu.store import store_configs as store_popu_configs


def store_configs(store: ZenStore) -> None:
    store(NeuroevolutionTaskConfig, name="config")
    store_algo_configs(store)
    store_eval_configs(store)
    store_popu_configs(store)
