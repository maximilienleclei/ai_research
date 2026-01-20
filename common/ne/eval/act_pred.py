"""Action Prediction evaluator for behavior cloning.

Shapes:
    NN: num_nets
    OD: obs_dim
    NA: n_actions
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium
import torch
from gymnasium import spaces
from jaxtyping import Bool, Float
from torch import Tensor

from common.ne.eval.base import BaseEval
from common.ne.popu.nets.dynamic.base import DynamicNets

if TYPE_CHECKING:
    from common.ne.popu.actor import ActorPopu


@dataclass
class ActPredEvalConfig:
    env_name: str
    target_agent_path: str  # Path to SB3 Zoo model (.zip)
    target_agent_algo: str  # "ppo", "sac", "td3", "a2c", "dqn"
    max_steps: int = 200
    num_workers: int = "${popu.config.size}"
    base_seed: int = "${config.seed}"


class ActPredEval(BaseEval):
    """Action prediction evaluator for behavior cloning.

    Each generation:
    1. Run target agent to collect (obs, action) trajectory
    2. Evaluate population on predicting target actions

    Fitness:
    - Discrete: soft accuracy (logit for correct action, then softmax)
    - Continuous: negative MSE
    """

    def __init__(self: "ActPredEval", config: ActPredEvalConfig) -> None:
        self.config = config
        env_name = config.env_name
        self.env = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make(env_name) for _ in range(config.num_workers)]
        )
        self._obs_space = self.env.single_observation_space
        self._action_space = self.env.single_action_space
        self._is_discrete = isinstance(self._action_space, spaces.Discrete)
        self.target_agent = self._load_target_agent()

    def _load_target_agent(self: "ActPredEval") -> Any:
        """Load SB3 model from zoo."""
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

        algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "dqn": DQN}
        algo_cls = algo_map.get(self.config.target_agent_algo.lower())
        if algo_cls is None:
            raise ValueError(
                f"Unknown algo: {self.config.target_agent_algo}. "
                f"Supported: {list(algo_map.keys())}"
            )
        return algo_cls.load(self.config.target_agent_path)

    def retrieve_num_inputs_outputs(self: "ActPredEval") -> tuple[int, int]:
        """Returns (obs_dim, n_actions)."""
        obs_dim = self._obs_space.shape[0]
        if self._is_discrete:
            n_actions = self._action_space.n
        else:
            n_actions = self._action_space.shape[0]
        return (obs_dim, n_actions)

    def retrieve_action_space(self: "ActPredEval") -> spaces.Space:
        """Returns the gymnasium action space."""
        return self._action_space

    def retrieve_input_output_specs(
        self: "ActPredEval",
    ) -> tuple[spaces.Space, spaces.Space]:
        """Returns (observation_space, action_space)."""
        return (self._obs_space, self._action_space)

    def _get_action_logits(
        self: "ActPredEval", population: "ActorPopu", x: Tensor
    ) -> Tensor:
        """Get raw action logits from population."""
        if isinstance(population.nets, DynamicNets):
            return population.nets(x)
        else:
            x = x.unsqueeze(1)
            logits = population.nets(x)
            return logits.squeeze(1)

    def __call__(
        self: "ActPredEval", population: "ActorPopu", generation: int = 0
    ) -> Float[Tensor, "NN"]:
        num_nets: int = population.config.size
        device = population.nets.config.device
        fitness: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        # Collect target agent trajectory
        population.nets.reset()
        obs_np, _ = self.env.reset(
            seed=[self.config.base_seed + generation] * num_nets
        )
        env_done: Bool[Tensor, "NN"] = torch.zeros(
            num_nets, dtype=torch.bool, device=device
        )
        env_rewards: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)
        steps: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        for step in range(self.config.max_steps):
            obs: Float[Tensor, "NN OD"] = torch.from_numpy(obs_np).float().to(device)

            # Get target agent's actions
            target_actions, _ = self.target_agent.predict(obs_np, deterministic=True)

            # Get population's action logits
            pred_logits = self._get_action_logits(population, obs)

            # Compute fitness contribution for active envs
            active = ~env_done
            if self._is_discrete:
                # Soft accuracy: softmax probability of correct action
                probs = torch.softmax(pred_logits, dim=-1)
                target_indices = torch.from_numpy(target_actions).long().to(device)
                correct_probs = probs.gather(1, target_indices.unsqueeze(1)).squeeze(1)
                fitness += correct_probs * active.float()
            else:
                # Negative MSE
                target_actions_t = torch.from_numpy(target_actions).float().to(device)
                # Map logits to action space (same as ActorPopu.map_actions)
                pred_actions = torch.tanh(pred_logits)
                low = torch.tensor(self._action_space.low, device=device).float()
                high = torch.tensor(self._action_space.high, device=device).float()
                pred_actions = (high + low) / 2 + pred_actions * (high - low) / 2
                mse = ((pred_actions - target_actions_t) ** 2).mean(dim=-1)
                fitness -= mse * active.float()

            steps += active.float()

            # Step environment with target agent's actions
            obs_np, rewards, terminated, truncated, _ = self.env.step(target_actions)
            env_rewards += torch.from_numpy(rewards).float().to(device) * active.float()
            done = terminated | truncated
            env_done = env_done | torch.from_numpy(done).to(device)

            if env_done.all():
                break

        # Normalize by number of steps
        fitness = fitness / steps.clamp(min=1)

        # Store for logging
        self._last_env_rewards = env_rewards

        return fitness

    def get_metrics(self: "ActPredEval") -> dict[str, Tensor]:
        """Return metrics from last evaluation.

        Returns:
            Metric name to tensor value mapping with keys:
            - env_rewards: Environment rewards from target agent rollouts
        """
        return {"env_rewards": self._last_env_rewards}
