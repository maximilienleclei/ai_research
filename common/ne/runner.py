"""Task runner for neuroevolution experiments.

This module provides the entry point for running neuroevolution tasks.
It handles Hydra configuration composition and delegates to the evolve() loop.

Usage
-----
From command line:
    python -m projects.{project} task={task}

The runner:
1. Loads configs from the project's task/ directory
2. Registers NE-specific config stores (algo, eval, popu, nets)
3. Instantiates components and calls evolve()
"""

from typing import Any

from hydra_zen import ZenStore

from common.ne.algo.base import BaseAlgo
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.evolve import evolve
from common.ne.popu.base import BasePopu
from common.ne.store import store_configs as store_ne_configs
from common.runner import BaseTaskRunner


class NeuroevolutionTaskRunner(BaseTaskRunner):
    """Task runner for neuroevolution experiments.

    Inherits from BaseTaskRunner and adds NE-specific configuration stores.
    Projects should subclass this and call run_task() from their __main__.py.

    Examples
    --------
    >>> class MyProjectRunner(NeuroevolutionTaskRunner):
    ...     project_path = Path(__file__).parent
    ...
    >>> if __name__ == "__main__":
    ...     MyProjectRunner.run_task()
    """

    @classmethod
    def store_configs(
        cls: type["NeuroevolutionTaskRunner"],
        store: ZenStore,
    ) -> None:
        """Register all NE config stores.

        Adds algorithm, evaluator, population, and network configs to the
        Hydra store for composition.
        """
        super().store_configs(store)
        store_ne_configs(store)

    @classmethod
    def run_subtask(
        cls: type["NeuroevolutionTaskRunner"],
        algo: BaseAlgo,
        eval: BaseEval,
        popu: BasePopu,
        config: NeuroevolutionSubtaskConfig,
    ) -> Any:
        """Run a single neuroevolution subtask.

        Args:
            algo: Selection algorithm instance.
            eval: Fitness evaluator instance.
            popu: Population instance with networks.
            config: Subtask configuration.

        Returns:
            Result from evolve() (currently None).
        """
        return evolve(algo, eval, popu, config)
