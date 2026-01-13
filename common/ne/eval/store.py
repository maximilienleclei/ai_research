from hydra_zen import ZenStore

from common.ne.eval.adv_gen import AdvGenEval, AdvGenEvalConfig
from common.ne.eval.score import ScoreEval, ScoreEvalConfig
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(ScoreEval, config=generate_config(ScoreEvalConfig)),
        name="score",
        group="eval",
    )
    store(
        generate_config(AdvGenEval, config=generate_config(AdvGenEvalConfig)),
        name="adv_gen",
        group="eval",
    )
