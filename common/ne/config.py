from dataclasses import dataclass, field
from typing import Any

from hydra_zen import make_config

from common.config import BaseSubtaskConfig
from common.ne.algo.base import BaseAlgo
from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu
from common.utils.hydra_zen import generate_config


@dataclass
class NeuroevolutionSubtaskConfig(BaseSubtaskConfig):
    num_minutes: int = 60


@dataclass
class NeuroevolutionTaskConfig(
    make_config(
        algo=generate_config(BaseAlgo),
        eval=generate_config(BaseEval),
        popu=generate_config(BasePopu),
        config=generate_config(NeuroevolutionSubtaskConfig),
    ),
):
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            "project",
            "task",
            {"task": None},
        ],
    )
