from hydra_zen import ZenStore

from common.ne.runner import NeuroevolutionTaskRunner
from common.utils.hydra_zen import generate_config
from projects.ne_control_score_encoded.eval import EncodedGymScoreEval, EncodedScoreEvalConfig


class TaskRunner(NeuroevolutionTaskRunner):
    """Task runner for neuroevolution with autoencoder-encoded states.

    Workflow:
    1. Collect states: python -m projects.ne_control_score_encoded.collect env_name=CartPole-v1
    2. Train autoencoder: python -m projects.ne_control_score_encoded.train task=cartpole
    3. Evolve: python -m projects.ne_control_score_encoded task=cartpole
    """

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store)
        store(
            generate_config(
                EncodedGymScoreEval,
                config=generate_config(EncodedScoreEvalConfig),
            ),
            name="encoded_score_gym",
            group="eval",
        )


TaskRunner.run_task()
