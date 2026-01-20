"""Configuration classes for neuroevolution experiments.

This module defines the Hydra configuration structure for neuroevolution tasks.
The configuration hierarchy is:

    NeuroevolutionTaskConfig
    ├── algo: BaseAlgo          # Selection algorithm (e.g., SimpleGA)
    ├── eval: BaseEval          # Fitness evaluator (e.g., GymScoreEval)
    ├── popu: BasePopu          # Population wrapper (e.g., ActorPopu)
    └── config: NeuroevolutionSubtaskConfig
        ├── seed: int           # Random seed
        ├── output_dir: str     # Results directory
        └── num_minutes: float  # Time budget for evolution
"""

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
    """Configuration for a single neuroevolution run.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility (inherited from BaseSubtaskConfig).
    output_dir : str
        Directory for results and checkpoints (inherited from BaseSubtaskConfig).
    num_minutes : float
        Time budget for evolution in minutes. The evolution loop runs until
        this wall-clock time is exceeded.
    """

    num_minutes: float = 60.0


@dataclass
class NeuroevolutionTaskConfig(
    make_config(
        algo=generate_config(BaseAlgo),
        eval=generate_config(BaseEval),
        popu=generate_config(BasePopu),
        config=generate_config(NeuroevolutionSubtaskConfig),
    ),
):
    """Top-level configuration for neuroevolution tasks.

    This config is composed from YAML files via Hydra. The `defaults` list
    specifies the order of config composition.

    Attributes
    ----------
    algo : BaseAlgo
        Selection algorithm configuration.
    eval : BaseEval
        Fitness evaluator configuration.
    popu : BasePopu
        Population configuration (includes network configuration).
    config : NeuroevolutionSubtaskConfig
        Subtask-level settings (seed, output_dir, num_minutes).
    defaults : list[Any]
        Hydra defaults list for config composition.
    """

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            "project",
            "task",
            {"task": None},
        ],
    )
