from hydra_zen import ZenStore
from common.ne.runner import NeuroevolutionTaskRunner
from utils.hydra_zen import (generate_config,
                             generate_config_partial_no_full_sig)

from .agent import GymAgent, GymAgentConfig
from .space import GymReinforcementSpace, GymReinforcementSpaceConfig


class TaskRunner(NeuroevolutionTaskRunner):

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)
        store(
            generate_config(
                GymReinforcementSpace,
                config=generate_config(GymReinforcementSpaceConfig),
            ),
            name="ne_control_score",
            group="space",
        )
        store(
            generate_config_partial_no_full_sig(
                GymAgent,
                config=generate_config(GymAgentConfig),
            ),
            name="ne_control_score",
            group="agent",
        )


TaskRunner.run_task()
