from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for human action prediction behavior cloning.

    Evolves networks to predict human actions from recorded behavior data.
    Supports optional continual learning features (session/run info).
    """

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)


TaskRunner.run_task()
