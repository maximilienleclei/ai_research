from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for action prediction behavior cloning.

    Evolves networks to predict target agent actions directly.
    Simpler than adversarial generation - no discriminator, just action matching.
    """

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)


TaskRunner.run_task()
