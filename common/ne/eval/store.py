"""Fitness evaluator configuration store registration.

This module registers evaluator configs under the `eval` group.

Available evaluators:
- score_gym: Environment rewards via Gymnasium (CPU-based envs)
- score_torchrl: Environment rewards via TorchRL (GPU-accelerated envs)
- act_pred: Action prediction accuracy (behavior cloning from SB3 agents)
- human_act_pred: Action prediction from human behavior data
- adv_gen: Adversarial generation (generator + discriminator fitness)

To add new evaluators:
1. Create class inheriting BaseEval in eval/
2. Implement __call__, retrieve_num_inputs_outputs, get_metrics
3. Register here with store(..., name="eval_name", group="eval")
4. Override in YAML: `override /eval: eval_name`
"""

from hydra_zen import ZenStore

from common.ne.eval.act_pred import ActPredEval, ActPredEvalConfig
from common.ne.eval.adv_gen import AdvGenEval, AdvGenEvalConfig
from common.ne.eval.human_act_pred import HumanActPredEval, HumanActPredEvalConfig
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.eval.score.gym import GymScoreEval
from common.ne.eval.score.torchrl import TorchRLScoreEval
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    """Register fitness evaluator configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    store(
        generate_config(GymScoreEval, config=generate_config(ScoreEvalConfig)),
        name="score_gym",
        group="eval",
    )
    store(
        generate_config(TorchRLScoreEval, config=generate_config(ScoreEvalConfig)),
        name="score_torchrl",
        group="eval",
    )
    store(
        generate_config(AdvGenEval, config=generate_config(AdvGenEvalConfig)),
        name="adv_gen",
        group="eval",
    )
    store(
        generate_config(ActPredEval, config=generate_config(ActPredEvalConfig)),
        name="act_pred",
        group="eval",
    )
    store(
        generate_config(HumanActPredEval, config=generate_config(HumanActPredEvalConfig)),
        name="human_act_pred",
        group="eval",
    )
