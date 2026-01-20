import gymnasium as gym
import torch

from common.ne.eval.base import BaseEval
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.popu.base import BasePopu


class GymScoreEval(BaseEval):
    """Score evaluator using Gym's AsyncVectorEnv (~50-100MB per worker)."""

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

    def retrieve_num_inputs_outputs(self: "GymScoreEval") -> tuple[int, int]:
        return (self._obs_shape[-1], self._num_actions)

    def retrieve_input_output_specs(
        self: "GymScoreEval",
    ) -> tuple[None, gym.spaces.Space]:
        return (None, self._action_space)

    def __call__(
        self: "GymScoreEval", population: BasePopu, generation: int = 0
    ) -> torch.Tensor:
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
            fitness_scores += torch.from_numpy(rewards).float().to(device) * (~env_done).float()
            env_done = env_done | torch.from_numpy(done).to(device)
        return fitness_scores
