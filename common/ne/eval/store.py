from hydra_zen import ZenStore

from common.ne.eval.act_pred import ActPredEval, ActPredEvalConfig
from common.ne.eval.adv_gen import AdvGenEval, AdvGenEvalConfig
from common.ne.eval.human_act_pred import HumanActPredEval, HumanActPredEvalConfig
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.eval.score.gym import GymScoreEval
from common.ne.eval.score.torchrl import TorchRLScoreEval
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
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
