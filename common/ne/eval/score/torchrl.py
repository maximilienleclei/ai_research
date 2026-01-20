"""TorchRL-based score evaluator for neuroevolution.

Uses TorchRL's ParallelEnv for parallel environment execution.
Higher memory footprint (~500MB per worker) but better GPU integration.
"""

import torch
from torch import Tensor
from torchrl.data import TensorSpec
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv

from common.ne.eval.base import BaseEval
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.popu.base import BasePopu


class TorchRLScoreEval(BaseEval):
    """Score evaluator using TorchRL's ParallelEnv.

    Evaluates network fitness based on cumulative environment rewards.
    Each network controls one parallel environment instance.

    Parameters
    ----------
    config : ScoreEvalConfig
        Configuration including env_name, max_steps, num_workers, seed, device.

    Attributes
    ----------
    env : ParallelEnv
        TorchRL parallel environment for execution.
    _last_env_rewards : Tensor
        Cached rewards from last evaluation for metrics reporting.

    Notes
    -----
    Memory usage is approximately 500MB per worker, higher than GymScoreEval.
    Use this when you need better GPU integration or TorchRL-specific features.
    """

    def __init__(self: "TorchRLScoreEval", config: ScoreEvalConfig) -> None:
        self.config = config
        env_name = config.env_name
        make_env = EnvCreator(lambda: GymEnv(env_name))
        self.env = ParallelEnv(config.num_workers, make_env)
        self._last_env_rewards: Tensor | None = None

    def retrieve_num_inputs_outputs(self: "TorchRLScoreEval") -> tuple[int, int]:
        """Return (observation_dim, action_dim) for network sizing."""
        return (
            self.env.observation_spec["observation"].shape[-1],
            self.env.action_spec.shape[-1],
        )

    def retrieve_input_output_specs(
        self: "TorchRLScoreEval",
    ) -> tuple[TensorSpec, TensorSpec]:
        """Return (observation_spec, action_spec) from TorchRL environment."""
        return (
            self.env.observation_spec["observation"],
            self.env.action_spec,
        )

    def __call__(
        self: "TorchRLScoreEval", population: BasePopu, generation: int = 0
    ) -> Tensor:
        """Evaluate population fitness via environment rollouts.

        Parameters
        ----------
        population : BasePopu
            Population of networks to evaluate.
        generation : int, optional
            Generation number for seeding (default: 0).

        Returns
        -------
        Tensor
            Fitness scores (cumulative rewards) with shape (num_nets,).
        """
        self.env.set_seed(self.config.seed + generation)
        num_envs = self.env.num_workers
        device = self.config.device
        fitness_scores = torch.zeros(num_envs, device=device)
        env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

        population.nets.reset()
        x = self.env.reset()

        for step in range(self.config.max_steps):
            x["action"] = population(x["observation"].to(device))
            x = self.env.step(x)["next"]
            reward = x["reward"].squeeze().to(device)
            fitness_scores += reward * (~env_done).float()
            env_done = env_done | x["done"].squeeze().to(device)

        self._last_env_rewards = fitness_scores
        return fitness_scores

    def get_metrics(self: "TorchRLScoreEval") -> dict[str, Tensor]:
        """Return environment rewards from last evaluation."""
        if self._last_env_rewards is None:
            return {}
        return {"env_rewards": self._last_env_rewards}
