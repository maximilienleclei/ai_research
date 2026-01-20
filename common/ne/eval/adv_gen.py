"""Shapes:

NN: num_nets
OD: obs_dim
NA: n_actions
K: num_disc_samples
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from jaxtyping import Bool, Float, Int
from torch import Tensor

from common.ne.eval.base import BaseEval

if TYPE_CHECKING:
    from common.ne.popu.adv_gen import AdvGenPopu


@dataclass
class AdvGenEvalConfig:
    env_name: str
    target_agent_path: str  # Path to SB3 Zoo model (.zip)
    target_agent_algo: str  # "ppo", "sac", "td3", "a2c", "dqn"
    max_steps: int = 200
    num_workers: int = "${popu.config.size}"
    num_disc_samples: int = 3  # Number of generators each discriminator evaluates
    base_seed: int = "${config.seed}"  # Base seed for env resets (gen i uses base_seed + i)


class AdvGenEval(BaseEval):
    """Adversarial evaluator for co-evolutionary imitation learning.

    Each network acts as both generator and discriminator:
    - As generator: produces trajectories, fitness = D(x_G)
    - As discriminator: evaluates trajectories, fitness = D(x_T) - D(x_G)

    Networks are randomly paired each generation.
    """

    def __init__(self: "AdvGenEval", config: AdvGenEvalConfig) -> None:
        self.config = config
        env_name = config.env_name
        self.env = gymnasium.vector.SyncVectorEnv(
            [lambda: gymnasium.make(env_name) for _ in range(config.num_workers)]
        )
        # Cache action/observation space info
        self._obs_space = self.env.single_observation_space
        self._action_space = self.env.single_action_space
        self._is_discrete = isinstance(self._action_space, spaces.Discrete)
        self.target_agent = self._load_target_agent()

    def _load_target_agent(self: "AdvGenEval") -> Any:
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

    def retrieve_num_inputs_outputs(self: "AdvGenEval") -> tuple[int, int]:
        """Returns (obs_dim, n_actions + 1) for dual-output networks."""
        obs_dim = self._obs_space.shape[0]
        if self._is_discrete:
            n_actions = self._action_space.n
        else:
            n_actions = self._action_space.shape[0]
        return (obs_dim, n_actions + 1)  # +1 for discriminator output

    def retrieve_action_space(self: "AdvGenEval") -> spaces.Space:
        """Returns the gymnasium action space."""
        return self._action_space

    def retrieve_input_output_specs(
        self: "AdvGenEval",
    ) -> tuple[spaces.Space, spaces.Space]:
        """Returns (observation_space, action_space)."""
        return (self._obs_space, self._action_space)

    def __call__(
        self: "AdvGenEval", population: "AdvGenPopu", generation: int = 0
    ) -> Float[Tensor, "NN"]:
        num_nets: int = population.config.size
        device = population.nets.config.device
        fitness_G: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)
        fitness_D: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        # Phase 1: Run all generators in parallel, collect trajectories
        population.nets.reset()
        obs_np, _ = self.env.reset(seed=[self.config.base_seed + generation] * num_nets)
        gen_trajectories: list[Float[Tensor, "NN OD"]] = []
        env_done: Bool[Tensor, "NN"] = torch.zeros(
            num_nets, dtype=torch.bool, device=device
        )
        env_rewards: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        for step in range(self.config.max_steps):
            obs: Float[Tensor, "NN OD"] = torch.from_numpy(obs_np).float().to(device)
            gen_trajectories.append(obs.clone())
            actions: Tensor = population.get_actions(obs)
            # Convert to numpy for gymnasium
            actions_np = actions.cpu().numpy()
            obs_np, rewards, terminated, truncated, _ = self.env.step(actions_np)
            env_rewards += torch.from_numpy(rewards).float().to(device) * (~env_done).float()
            done = terminated | truncated
            env_done: Bool[Tensor, "NN"] = (
                env_done | torch.from_numpy(done).to(device)
            )

        # Phase 2: Compute D(x_G) - each discriminator evaluates K random generators
        # disc_to_gen[j, k] = which generator discriminator j evaluates for sample k
        K: int = self.config.num_disc_samples
        disc_to_gen: Int[Tensor, "NN K"] = torch.randint(
            num_nets, (num_nets, K), device=device
        )
        gen_eval_counts: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)
        D_x_G_sum: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        for k in range(K):
            population.nets.reset()
            gen_indices: Int[Tensor, "NN"] = disc_to_gen[:, k]
            sample_score_sum: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

            for obs in gen_trajectories:
                # disc j sees obs from gen gen_indices[j]
                obs_for_disc: Float[Tensor, "NN OD"] = obs[gen_indices]
                disc_scores: Float[Tensor, "NN"] = population.get_discrimination(
                    obs_for_disc
                )
                # Accumulate scores to generators
                fitness_G.scatter_add_(0, gen_indices, disc_scores)
                sample_score_sum: Float[Tensor, "NN"] = sample_score_sum + disc_scores

            # Track evaluations per generator and D(x_G) per discriminator
            ones: Float[Tensor, "NN"] = torch.ones(num_nets, device=device)
            gen_eval_counts.scatter_add_(0, gen_indices, ones)
            D_x_G_sum: Float[Tensor, "NN"] = (
                D_x_G_sum + sample_score_sum / len(gen_trajectories)
            )

        # Phase 3: Run target agent in parallel envs, each discriminator sees different run
        population.nets.reset()
        obs_T_np, _ = self.env.reset(seed=[self.config.base_seed + generation] * num_nets)
        steps_T: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)
        done_T: Bool[Tensor, "NN"] = torch.zeros(
            num_nets, dtype=torch.bool, device=device
        )
        target_env_rewards: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        for step in range(self.config.max_steps):
            obs_T: Float[Tensor, "NN OD"] = torch.from_numpy(obs_T_np).float().to(device)
            actions_T, _ = self.target_agent.predict(obs_T_np, deterministic=True)
            obs_T_np, rewards_T, terminated, truncated, _ = self.env.step(actions_T)
            done_np = terminated | truncated

            # Track target agent's actual env rewards
            target_env_rewards += (
                torch.from_numpy(rewards_T).float().to(device) * (~done_T).float()
            )

            # Evaluate with discriminators (only active envs contribute)
            disc_scores_T: Float[Tensor, "NN"] = population.get_discrimination(obs_T)
            fitness_D: Float[Tensor, "NN"] = (
                fitness_D + disc_scores_T * (~done_T).float()
            )
            steps_T: Float[Tensor, "NN"] = steps_T + (~done_T).float()

            done_T: Bool[Tensor, "NN"] = done_T | torch.from_numpy(done_np).to(device)
            if done_T.all():
                break

        # Normalize generator fitness by (num_evals * trajectory_length)
        fitness_G: Float[Tensor, "NN"] = fitness_G / (
            gen_eval_counts * len(gen_trajectories)
        ).clamp(min=1)
        # Normalize discriminator's target score by trajectory length
        fitness_D: Float[Tensor, "NN"] = fitness_D / steps_T.clamp(min=1)

        # Discriminator fitness = D(x_T) - D(x_G)
        # D_x_G_sum is already averaged over trajectory length, just need to avg over K
        D_x_G_for_disc: Float[Tensor, "NN"] = D_x_G_sum / K
        fitness_D: Float[Tensor, "NN"] = fitness_D - D_x_G_for_disc

        # Store fitness components and env rewards for logging
        self._last_fitness_G = fitness_G
        self._last_fitness_D = fitness_D
        self._last_env_rewards = env_rewards
        self._last_target_env_rewards = target_env_rewards

        # Combined fitness: each network's total = generator fitness + discriminator fitness
        return fitness_G + fitness_D

    def get_metrics(self: "AdvGenEval") -> dict[str, Tensor]:
        """Return adversarial training metrics from last evaluation.

        Returns
        -------
        dict[str, Tensor]
            - fitness_G: Generator fitness scores
            - fitness_D: Discriminator fitness scores
            - env_rewards: Environment rewards from generator rollouts
            - target_env_rewards: Environment rewards from target agent rollouts
        """
        return {
            "fitness_G": self._last_fitness_G,
            "fitness_D": self._last_fitness_D,
            "env_rewards": self._last_env_rewards,
            "target_env_rewards": self._last_target_env_rewards,
        }
