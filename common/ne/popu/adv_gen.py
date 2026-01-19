"""Shapes:

NN: num_nets
OD: obs_dim
NA: n_actions
NO: num_outputs (n_actions + 1)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from jaxtyping import Float, Int
from torch import Tensor

from common.ne.popu.base import BasePopu, BasePopuConfig
from common.ne.popu.nets.dynamic.base import DynamicNets

if TYPE_CHECKING:
    from common.ne.eval.adv_gen import AdvGenEval


@dataclass
class AdvGenPopuConfig(BasePopuConfig):
    eval: "AdvGenEval" = "${eval}"

    def __post_init__(self: "AdvGenPopuConfig") -> None:
        action_space = self.eval.retrieve_action_space()
        self._is_discrete = isinstance(action_space, spaces.Discrete)
        if self._is_discrete:
            self.n_actions = action_space.n
        else:
            self.n_actions = action_space.shape[0]
            self._action_low = torch.from_numpy(action_space.low).float()
            self._action_high = torch.from_numpy(action_space.high).float()
        delattr(self, "eval")


class AdvGenPopu(BasePopu):
    """Population that can both generate actions and discriminate.

    Each network has n_actions + 1 outputs:
    - First n_actions outputs: action generation
    - Last output: discrimination score [0,1]
    """

    config: AdvGenPopuConfig

    def _forward(
        self: "AdvGenPopu", x: Float[Tensor, "NN OD"]
    ) -> Float[Tensor, "NN NO"]:
        """Get raw network outputs (n_actions + 1)."""
        if isinstance(self.nets, DynamicNets):
            return self.nets(x)
        else:
            x: Float[Tensor, "NN 1 OD"] = x.unsqueeze(1)
            outputs: Float[Tensor, "NN 1 NO"] = self.nets(x)
            return outputs.squeeze(1)

    def get_actions(
        self: "AdvGenPopu", x: Float[Tensor, "NN OD"]
    ) -> Int[Tensor, "NN"] | Float[Tensor, "NN NA"]:
        """Extract action outputs and convert to env format."""
        outputs: Float[Tensor, "NN NO"] = self._forward(x)
        action_logits: Float[Tensor, "NN NA"] = outputs[:, : self.config.n_actions]

        if self.config._is_discrete:
            # Gymnasium expects action indices for discrete spaces
            action_indices: Int[Tensor, "NN"] = torch.argmax(action_logits, dim=-1)
            return action_indices
        else:
            # Scale tanh output to action bounds
            tanh_actions: Float[Tensor, "NN NA"] = torch.tanh(action_logits)
            low = self.config._action_low.to(x.device)
            high = self.config._action_high.to(x.device)
            return (high + low) / 2 + tanh_actions * (high - low) / 2

    def get_discrimination(
        self: "AdvGenPopu", x: Float[Tensor, "NN OD"]
    ) -> Float[Tensor, "NN"]:
        """Extract discrimination output (last), squash to [0,1]."""
        outputs: Float[Tensor, "NN NO"] = self._forward(x)
        disc_score: Float[Tensor, "NN"] = outputs[:, -1]
        return torch.sigmoid(disc_score)

    def __call__(
        self: "AdvGenPopu", x: Float[Tensor, "NN OD"]
    ) -> Int[Tensor, "NN"] | Float[Tensor, "NN NA"]:
        """Default: return actions (for compatibility with existing code)."""
        return self.get_actions(x)
