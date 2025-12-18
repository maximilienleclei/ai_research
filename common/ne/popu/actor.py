import torch
import torch.nn.functional as F

from common.ne.popu.base import BasePopu


class ActorPopu(BasePopu):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (num_envs, obs_dim) where num_envs == num_nets
        # nets expects (num_nets, batch_size, obs_dim)
        x = x.unsqueeze(1)  # (num_nets, 1, obs_dim)
        action_logits = self.nets(x)  # (num_nets, 1, num_actions)
        action_logits = action_logits.squeeze(1)  # (num_nets, num_actions)
        action_indices = torch.argmax(action_logits, dim=-1)  # (num_nets,)
        # TorchRL GymEnv expects one-hot actions
        actions = F.one_hot(action_indices, num_classes=action_logits.shape[-1])
        return actions
