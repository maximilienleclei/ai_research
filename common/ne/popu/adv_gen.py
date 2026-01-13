from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from common.ne.popu.base import BasePopu, BasePopuConfig
from common.ne.popu.nets.dynamic.base import DynamicNets

if TYPE_CHECKING:
    from common.ne.eval.adv_gen import AdvGenEval


@dataclass
class AdvGenPopuConfig(BasePopuConfig):
    eval: "AdvGenEval" = "${eval}"

    def __post_init__(self: "AdvGenPopuConfig") -> None:
        _, self.action_spec = self.eval.retrieve_input_output_specs()
        self.n_actions = self.action_spec.shape[-1]
        delattr(self, "eval")


class AdvGenPopu(BasePopu):
    """Population that can both generate actions and discriminate.

    Each network has n_actions + 1 outputs:
    - First n_actions outputs: action generation
    - Last output: discrimination score [0,1]
    """

    config: AdvGenPopuConfig

    def _forward(self: "AdvGenPopu", x: Tensor) -> Tensor:
        """Get raw network outputs (n_actions + 1)."""
        if isinstance(self.nets, DynamicNets):
            # DynamicNets expects (num_nets, obs_dim), no batch dim
            return self.nets(x)  # (num_nets, n_actions + 1)
        else:
            # Static nets expect (num_nets, batch_size, obs_dim)
            x = x.unsqueeze(1)  # (num_nets, 1, obs_dim)
            outputs = self.nets(x)  # (num_nets, 1, n_actions + 1)
            return outputs.squeeze(1)  # (num_nets, n_actions + 1)

    def get_actions(self: "AdvGenPopu", x: Tensor) -> Tensor:
        """Extract action outputs and convert to env format."""
        outputs = self._forward(x)
        action_logits = outputs[:, : self.config.n_actions]

        if self.config.action_spec.domain == "discrete":
            action_indices = torch.argmax(action_logits, dim=-1)
            return F.one_hot(action_indices, num_classes=self.config.n_actions)
        else:
            tanh_actions = torch.tanh(action_logits)
            low = self.config.action_spec.space.low
            high = self.config.action_spec.space.high
            return (high + low) / 2 + tanh_actions * (high - low) / 2

    def get_discrimination(self: "AdvGenPopu", x: Tensor) -> Tensor:
        """Extract discrimination output (last), clip to [0,1]."""
        outputs = self._forward(x)
        disc_score = outputs[:, -1]  # Last output
        return torch.clamp(torch.relu(disc_score), 0, 1)

    def __call__(self: "AdvGenPopu", x: Tensor) -> Tensor:
        """Default: return actions (for compatibility with existing code)."""
        return self.get_actions(x)
