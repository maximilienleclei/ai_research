"""Base class for neuroevolution evaluators.

Evaluators compute fitness scores for a population of networks by evaluating
their performance on a specific task (environment control, behavior cloning,
adversarial generation, etc.).

Architecture
------------
Evaluators follow a simple contract:
1. __call__(population, generation) -> fitness_scores tensor
2. retrieve_num_inputs_outputs() -> (obs_dim, action_dim) for network sizing
3. retrieve_input_output_specs() -> (obs_spec, action_spec) for action processing
4. get_metrics() -> dict of additional metrics from last evaluation (optional)

The get_metrics() pattern replaces ad-hoc attribute checking (hasattr) with a
standard interface for logging additional evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from common.ne.popu.base import BasePopu


class BaseEval(ABC):
    """Abstract base class for population fitness evaluators.

    Subclasses must implement the evaluation logic and provide information
    about the input/output dimensions for network construction.

    Attributes
    ----------
    config : object
        Evaluator-specific configuration (defined in subclasses).
    """

    @abstractmethod
    def __call__(
        self: "BaseEval", population: "BasePopu", generation: int = 0
    ) -> Tensor:
        """Evaluate all networks in the population and return fitness scores.

        Parameters
        ----------
        population : BasePopu
            Population of networks to evaluate.
        generation : int, optional
            Current generation number, used for seeding (default: 0).

        Returns
        -------
        Tensor
            Fitness scores with shape (num_nets,). Higher is better.
        """
        ...

    @abstractmethod
    def retrieve_num_inputs_outputs(self: "BaseEval") -> tuple[int, int]:
        """Return the network input and output dimensions.

        Returns
        -------
        tuple[int, int]
            (num_inputs, num_outputs) - observation dim and action dim.
        """
        ...

    @abstractmethod
    def retrieve_input_output_specs(self: "BaseEval") -> tuple[Any, Any]:
        """Return the observation and action space specifications.

        Returns
        -------
        tuple[Any, Any]
            (obs_spec, action_spec) - gymnasium spaces or TorchRL specs.
        """
        ...

    def get_metrics(self: "BaseEval") -> dict[str, Tensor]:
        """Return additional metrics from the last evaluation.

        Override this method to expose evaluation-specific metrics for logging.
        The evolve() loop will call this after each evaluation to log metrics.

        Returns
        -------
        dict[str, Tensor]
            Metric name -> tensor value mapping. Common keys include:
            - "fitness_G": Generator fitness (adversarial evaluators)
            - "fitness_D": Discriminator fitness (adversarial evaluators)
            - "env_rewards": Raw environment rewards (score-based evaluators)
            - "target_env_rewards": Target agent rewards (imitation evaluators)

        Examples
        --------
        >>> def get_metrics(self):
        ...     return {
        ...         "env_rewards": self._last_env_rewards,
        ...         "episode_lengths": self._last_episode_lengths,
        ...     }
        """
        return {}
