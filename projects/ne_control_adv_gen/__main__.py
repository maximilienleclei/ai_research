from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for adversarial generation imitation learning.

    Uses co-evolutionary framework where networks both generate behavior
    and discriminate between generated and target agent trajectories.
    """

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)


TaskRunner.run_task()
