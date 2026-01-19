from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    eval: "BaseEval" = "${eval}"

    def __post_init__(self: "ActorPopuConfig") -> None:
        _, self.action_spec = self.eval.retrieve_input_output_specs()
        delattr(self, "eval")


class ActorPopu(BasePopu):

    def _is_discrete(self: "ActorPopu") -> bool:
        """Check if action space is discrete (handles both Gym and TorchRL specs)."""
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
        """Forward pass through network to get raw action logits."""
        if isinstance(self.nets, DynamicNets):
            # DynamicNets expects (num_nets, obs_dim), no batch dim
            return self.nets(x)  # (num_nets, num_actions)
        else:
            # Static nets expect (num_nets, batch_size, obs_dim)
            x = x.unsqueeze(1)  # (num_nets, 1, obs_dim)
            action_logits = self.nets(x)  # (num_nets, 1, num_actions)
            return action_logits.squeeze(1)  # (num_nets, num_actions)

    def discretize_actions(self: "ActorPopu", action_logits: Tensor) -> Tensor:
        """Convert logits to one-hot discrete actions."""
        action_indices = torch.argmax(action_logits, dim=-1)  # (num_nets,)
        return F.one_hot(action_indices, num_classes=action_logits.shape[-1])

    def map_actions(self: "ActorPopu", action_logits: Tensor) -> Tensor:
        """Convert logits to bounded continuous actions via tanh + rescale."""
        tanh_actions = torch.tanh(action_logits)  # (num_nets, num_actions) in [-1, 1]
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
        # x is (num_envs, obs_dim) where num_envs == num_nets
        action_logits = self.get_action_logits(x)

        if self._is_discrete():
            return self.discretize_actions(action_logits)
        else:
            return self.map_actions(action_logits)
