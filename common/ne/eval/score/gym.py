"""Gymnasium-based score evaluator for neuroevolution.

Uses Gymnasium's SyncVectorEnv for parallel environment execution.
Memory footprint is relatively low (~50-100MB per worker).
"""

import gymnasium as gym
import torch
from torch import Tensor

from common.ne.eval.base import BaseEval
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.popu.base import BasePopu


class GymScoreEval(BaseEval):
    """Score evaluator using Gymnasium's SyncVectorEnv.

    Evaluates network fitness based on cumulative environment rewards.
    Each network controls one parallel environment instance.

    Parameters
    ----------
    config : ScoreEvalConfig
        Configuration including env_name, max_steps, num_workers, seed, device.

    Attributes
    ----------
    env : gym.vector.SyncVectorEnv
        Vectorized environment for parallel execution.
    _last_env_rewards : Tensor
        Cached rewards from last evaluation for metrics reporting.

    Notes
    -----
    Memory usage is approximately 50-100MB per worker, making this suitable
    for large population sizes. For heavier environments, consider TorchRLScoreEval.
    """

    def __init__(self: "GymScoreEval", config: ScoreEvalConfig) -> None:
        self.config = config
        env_name = config.env_name
        self.env = gym.vector.SyncVectorEnv(
            [lambda en=env_name: gym.make(en) for _ in range(config.num_workers)]
        )
        # Cache action space info
        single_env = gym.make(env_name)
        self._obs_shape = single_env.observation_space.shape
        self._action_space = single_env.action_space
        if isinstance(self._action_space, gym.spaces.Discrete):
            self._num_actions = self._action_space.n
            self._is_discrete = True
        else:
            self._num_actions = self._action_space.shape[-1]
            self._is_discrete = False
        single_env.close()
        self._last_env_rewards: Tensor | None = None

    def retrieve_num_inputs_outputs(self: "GymScoreEval") -> tuple[int, int]:
        """Return (observation_dim, action_dim) for network sizing."""
        return (self._obs_shape[-1], self._num_actions)

    def retrieve_input_output_specs(
        self: "GymScoreEval",
    ) -> tuple[None, gym.spaces.Space]:
        """Return (None, action_space) - obs_spec not used for Gym envs."""
        return (None, self._action_space)

    def __call__(
        self: "GymScoreEval", population: BasePopu, generation: int = 0
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
        num_envs = self.config.num_workers
        device = self.config.device
        fitness_scores = torch.zeros(num_envs, device=device)
        env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

        population.nets.reset()
        obs, _ = self.env.reset(seed=[self.config.seed + generation] * num_envs)
        obs = torch.from_numpy(obs).float().to(device)

        for step in range(self.config.max_steps):
            actions = population(obs)
            if self._is_discrete:
                actions_np = actions.argmax(dim=-1).cpu().numpy()
            else:
                actions_np = actions.cpu().numpy()
            obs, rewards, terminated, truncated, _ = self.env.step(actions_np)
            obs = torch.from_numpy(obs).float().to(device)
            done = terminated | truncated
            fitness_scores += (
                torch.from_numpy(rewards).float().to(device) * (~env_done).float()
            )
            env_done = env_done | torch.from_numpy(done).to(device)

        self._last_env_rewards = fitness_scores
        return fitness_scores

    def get_metrics(self: "GymScoreEval") -> dict[str, Tensor]:
        """Return environment rewards from last evaluation."""
        if self._last_env_rewards is None:
            return {}
        return {"env_rewards": self._last_env_rewards}
