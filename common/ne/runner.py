from typing import Any

from hydra_zen import ZenStore

from common.ne.algo.base import BaseAlgo
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.evolve import evolve
from common.ne.popu.base import BasePopu
from common.ne.store import store_configs as store_ne_configs
from common.runner import BaseTaskRunner


class NeuroevolutionTaskRunner(BaseTaskRunner):
    @classmethod
    def store_configs(
        cls: type["NeuroevolutionTaskRunner"],
        store: ZenStore,
    ) -> None:
        super().store_configs(store)
        store_ne_configs(store)

    @classmethod
    def run_subtask(
        cls: type["NeuroevolutionTaskRunner"],
        algo: BaseAlgo,
        eval: BaseEval,
        popu: BasePopu,
        config: NeuroevolutionSubtaskConfig,
    ) -> Any:
        return evolve(algo, eval, popu, config)
