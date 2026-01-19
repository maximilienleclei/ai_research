import torch
from torchrl.data import TensorSpec
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv

from common.ne.eval.base import BaseEval
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.popu.base import BasePopu


class TorchRLScoreEval(BaseEval):
    """Score evaluator using TorchRL's ParallelEnv (~500MB per worker)."""

    def __init__(self: "TorchRLScoreEval", config: ScoreEvalConfig) -> None:
        self.config = config
        env_name = config.env_name
        make_env = EnvCreator(lambda: GymEnv(env_name))
        self.env = ParallelEnv(config.num_workers, make_env)

    def retrieve_num_inputs_outputs(self: "TorchRLScoreEval") -> tuple[int, int]:
        return (
            self.env.observation_spec["observation"].shape[-1],
            self.env.action_spec.shape[-1],
        )

    def retrieve_input_output_specs(
        self: "TorchRLScoreEval",
    ) -> tuple[TensorSpec, TensorSpec]:
        return (
            self.env.observation_spec["observation"],
            self.env.action_spec,
        )

    def __call__(
        self: "TorchRLScoreEval", population: BasePopu, generation: int = 0
    ) -> torch.Tensor:
        self.env.set_seed(self.config.seed + generation)
        num_envs = self.env.num_workers
        fitness_scores = torch.zeros(num_envs)
        env_done = torch.zeros(num_envs, dtype=torch.bool)
        population.nets.reset()
        x = self.env.reset()
        for step in range(self.config.max_steps):
            x["action"] = population(x["observation"])
            x = self.env.step(x)["next"]
            reward = x["reward"].squeeze()
            fitness_scores += reward * (~env_done).float()
            env_done = env_done | x["done"].squeeze()
        return fitness_scores
