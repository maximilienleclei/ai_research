from dataclasses import dataclass

import torch
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv

from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu


@dataclass
class ScoreEvalConfig:
    env_name: str
    max_steps: int = 50
    num_workers: int = "${popu.config.size}"


class ScoreEval(BaseEval):
    def __init__(self, config: ScoreEvalConfig):
        self.config = config
        env_name = (
            config.env_name
        )  # Capture locally to avoid circular reference
        make_env = EnvCreator(lambda: GymEnv(env_name))
        self.env = ParallelEnv(config.num_workers, make_env)

    def retrieve_num_inputs_outputs(self) -> tuple[int, int]:
        return (
            self.env.observation_spec["observation"].shape[1],
            self.env.action_spec.space.n,
        )

    def __call__(self, population: BasePopu) -> torch.Tensor:
        num_envs = self.env.num_workers
        fitness_scores = torch.zeros(num_envs)
        x = self.env.reset()
        for step in range(self.config.max_steps):
            x["action"] = population(x["observation"])
            x = self.env.step(x)["next"]
            fitness_scores += x["reward"].squeeze()
        return fitness_scores
