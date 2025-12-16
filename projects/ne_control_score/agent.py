from dataclasses import dataclass
from typing import Annotated as An

import torch
import torch.nn.functional as f
from jaxtyping import Float32, Int64
from torch import Tensor
from torchrl.data.tensor_specs import ContinuousBox
from torchrl.envs.libs.gym import GymEnv

from common.ne.agent import BaseAgent, BaseAgentConfig
from common.ne.net.cpu.static import CPUStaticRNNFC, CPUStaticRNNFCConfig
from common.utils.beartype import ge, le, one_of
from common.utils.torch import RunningStandardization


@dataclass
class GymAgentConfig(BaseAgentConfig):

    env_name: str = "${space.config.env_name}"
    hidden_size: int = 50
    mutation_std: float = 0.01


class GymAgent(BaseAgent):

    def __init__(
        self: "GymAgent",
        config: GymAgentConfig,
        pop_idx: An[int, ge(0), le(1)],
        *,
        pops_are_merged: bool,
    ) -> None:
        super().__init__(
            config=config,
            pop_idx=pop_idx,
            pops_are_merged=pops_are_merged,
        )
        self.config: GymAgentConfig
        temp_env = GymEnv(env_name=config.env_name)
        self.num_actions = temp_env.action_spec.shape.numel()
        self.net = CPUStaticRNNFC(
            config=CPUStaticRNNFCConfig(
                input_size=temp_env.observation_spec[
                    "observation"
                ].shape.numel(),
                hidden_size=config.hidden_size,
                output_size=self.num_actions,
            ),
        )
        self.output_mode: An[
            str,
            one_of("continuous", "discrete"),
        ] = temp_env.action_spec.domain
        if self.output_mode == "continuous":
            action_space: ContinuousBox = temp_env.action_spec.space
            self.output_low = action_space.low
            self.output_high = action_space.high
        self.standardizer = RunningStandardization(self.net.rnn.input_size)

    def mutate(self: "GymAgent") -> None:
        for param in self.net.parameters():
            param.data += self.config.mutation_std * torch.randn_like(
                input=param.data,
            )

    def reset(self: "GymAgent") -> None:
        self.net.reset()

    def __call__(
        self: "GymAgent",
        x: Float32[Tensor, " num_obs"],
    ) -> Float32[Tensor, " num_actions"] | Int64[Tensor, " num_actions"]:
        x: Float32[Tensor, " num_obs"] = self.env_to_net(x=x)
        x: Float32[Tensor, " num_actions"] = self.net(x=x)
        x: Float32[Tensor, " num_actions"] | Int64[Tensor, " num_actions"] = (
            self.net_to_env(x=x)
        )
        return x

    def env_to_net(
        self: "GymAgent",
        x: Float32[Tensor, " num_obs"],
    ) -> Float32[Tensor, " out_size"]:
        x: Float32[Tensor, " num_obs"] = self.standardizer(x=x)
        return x

    def net_to_env(
        self: "GymAgent",
        x: Float32[Tensor, " num_actions"],
    ) -> Float32[Tensor, " num_actions"] | Int64[Tensor, " num_actions"]:
        if self.output_mode == "discrete":
            x: Float32[Tensor, " num_actions"] = torch.softmax(input=x, dim=0)
            x: Int64[Tensor, " "] = torch.multinomial(
                input=x,
                num_samples=1,
            ).squeeze()
            # Turn the integer into a one-hot vector.
            x: Int64[Tensor, " num_actions"] = f.one_hot(
                x,
                num_classes=self.num_actions,
            )
            return x
        else:
            x: Float32[Tensor, " num_actions"] = torch.tanh(input=x)
            x: Float32[Tensor, " num_actions"] = (
                x * (self.output_high - self.output_low) / 2
                + (self.output_high + self.output_low) / 2
            )
            return x
