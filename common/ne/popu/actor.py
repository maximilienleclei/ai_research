"""Actor population for action-taking networks.

This module defines the ActorPopu class, which wraps network populations to
provide action generation for reinforcement learning environments.

The population handles both discrete and continuous action spaces, mapping
raw network outputs (logits) to valid actions.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import Tensor

from common.ne.popu.base import BasePopu, BasePopuConfig
from common.ne.popu.nets.dynamic.base import DynamicNets

if TYPE_CHECKING:
    from common.ne.eval.base import BaseEval


@dataclass
class ActorPopuConfig(BasePopuConfig):
    """Configuration for actor populations.

    Attributes
    ----------
    size : int
        Number of actors in the population (inherited from BasePopuConfig).
    action_spec : Any
        Action space specification (gym.spaces or TorchRL spec).
        Computed from evaluator in __post_init__.

    Notes
    -----
    The `eval` field is used only during initialization to extract the action
    space specification. It is deleted after extraction to prevent circular
    references during serialization.
    """

    eval: "BaseEval" = field(default="${eval}", repr=False)  # type: ignore[assignment]
    # Computed field - set in __post_init__
    action_spec: Any = field(init=False, default=None)

    def __post_init__(self: "ActorPopuConfig") -> None:
        """Extract action spec from evaluator and clean up reference."""
        _, self.action_spec = self.eval.retrieve_input_output_specs()
        # Remove eval reference to prevent circular serialization
        del self.eval


class ActorPopu(BasePopu):
    """Population of actor networks that take actions in environments.

    Wraps network populations to provide action generation, handling both
    discrete and continuous action spaces.

    Attributes
    ----------
    config : ActorPopuConfig
        Configuration with population size and action space.
    nets : BaseNets
        Underlying network population.

    Examples
    --------
    >>> # Discrete action space (e.g., CartPole)
    >>> actions = actor_popu(observations)  # Returns one-hot encoded actions
    >>> action_indices = actions.argmax(dim=-1)

    >>> # Continuous action space (e.g., HalfCheetah)
    >>> actions = actor_popu(observations)  # Returns bounded actions
    """

    def _is_discrete(self: "ActorPopu") -> bool:
        """Check if action space is discrete.

        Handles both Gymnasium spaces and TorchRL specs.

        Returns
        -------
        bool
            True if action space is discrete, False if continuous.
        """
        spec = self.config.action_spec
        # Gym spaces
        if isinstance(spec, gym.spaces.Discrete):
            return True
        if isinstance(spec, gym.spaces.Box):
            return False
        # TorchRL specs
        if hasattr(spec, "domain"):
            return spec.domain == "discrete"
        return False

    def get_action_logits(self: "ActorPopu", x: Tensor) -> Tensor:
        """Forward pass through network to get raw action logits.

        Parameters
        ----------
        x : Tensor
            Observations with shape (num_nets, obs_dim).

        Returns
        -------
        Tensor
            Action logits with shape (num_nets, num_actions).
        """
        if isinstance(self.nets, DynamicNets):
            # DynamicNets expects (num_nets, obs_dim), no batch dim
            return self.nets(x)
        else:
            # Static nets expect (num_nets, batch_size, obs_dim)
            x = x.unsqueeze(1)
            action_logits = self.nets(x)
            return action_logits.squeeze(1)

    def discretize_actions(self: "ActorPopu", action_logits: Tensor) -> Tensor:
        """Convert logits to one-hot discrete actions.

        Parameters
        ----------
        action_logits : Tensor
            Raw network outputs with shape (num_nets, num_actions).

        Returns
        -------
        Tensor
            One-hot encoded actions with shape (num_nets, num_actions).
        """
        action_indices = torch.argmax(action_logits, dim=-1)
        return F.one_hot(action_indices, num_classes=action_logits.shape[-1])

    def map_actions(self: "ActorPopu", action_logits: Tensor) -> Tensor:
        """Convert logits to bounded continuous actions.

        Uses tanh squashing and linear rescaling to map unbounded network
        outputs to the action space bounds.

        Parameters
        ----------
        action_logits : Tensor
            Raw network outputs with shape (num_nets, num_actions).

        Returns
        -------
        Tensor
            Bounded actions with shape (num_nets, num_actions).
        """
        tanh_actions = torch.tanh(action_logits)  # [-1, 1]
        spec = self.config.action_spec

        # Handle both Gym Box and TorchRL specs
        if isinstance(spec, gym.spaces.Box):
            low, high = spec.low, spec.high
        else:
            low, high = spec.space.low, spec.space.high

        low = torch.from_numpy(low).float()
        high = torch.from_numpy(high).float()

        # Rescale from [-1, 1] to [low, high]
        return (high + low) / 2 + tanh_actions * (high - low) / 2

    def __call__(self: "ActorPopu", x: Tensor) -> Tensor:
        """Generate actions for given observations.

        Parameters
        ----------
        x : Tensor
            Observations with shape (num_nets, obs_dim).

        Returns
        -------
        Tensor
            Actions with shape (num_nets, num_actions).
            - Discrete: one-hot encoded
            - Continuous: bounded to action space
        """
        action_logits = self.get_action_logits(x)

        if self._is_discrete():
            return self.discretize_actions(action_logits)
        else:
            return self.map_actions(action_logits)
