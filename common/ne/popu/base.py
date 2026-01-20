"""Base classes for population wrappers in neuroevolution.

Populations wrap network collections and provide higher-level interfaces for
evaluation. While networks output raw logits, populations handle action space
mapping, discretization, and other task-specific processing.

Architecture::

    Population (BasePopu)
    └── Networks (BaseNets)
        └── Individual nets with weights

The population delegates forward passes to its networks but may add
post-processing (e.g., action discretization for RL environments).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from common.ne.popu.nets.base import BaseNets


@dataclass
class BasePopuConfig:
    """Configuration for population wrappers.

    Attributes:
        size: Number of individuals (networks) in the population.
    """

    size: int


class BasePopu(ABC):
    """Abstract base class for population wrappers.

    Populations provide a high-level interface for interacting with collections
    of networks. They handle task-specific processing like action mapping.

    Attributes:
        config: Population configuration.
        nets: Underlying network collection.

    Subclasses:
        - ActorPopu: Action-taking population for RL environments
        - AdvGenPopu: Dual-function population for adversarial generation
    """

    def __init__(self: "BasePopu", config: BasePopuConfig, nets: BaseNets) -> None:
        """Initialize population with configuration and networks.

        Args:
            config: Population configuration.
            nets: Network collection to wrap.
        """
        self.config = config
        self.nets = nets

    @abstractmethod
    def __call__(self: "BasePopu", x: Tensor) -> Tensor:
        """Process observations and return actions.

        Args:
            x: Observations with shape (num_nets, obs_dim).

        Returns:
            Actions with shape (num_nets, action_dim).
            Format depends on the subclass (e.g., one-hot for discrete).
        """
        ...
