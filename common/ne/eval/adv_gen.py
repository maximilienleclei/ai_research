from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torchrl.data import TensorSpec
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv

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
        make_env = EnvCreator(lambda: GymEnv(env_name))
        self.env = ParallelEnv(config.num_workers, make_env)
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
        obs_dim = self.env.observation_spec["observation"].shape[-1]
        n_actions = self.env.action_spec.shape[-1]
        return (obs_dim, n_actions + 1)  # +1 for discriminator output

    def retrieve_input_output_specs(
        self: "AdvGenEval",
    ) -> tuple[TensorSpec, TensorSpec]:
        return (
            self.env.observation_spec["observation"],
            self.env.action_spec,
        )

    def __call__(self: "AdvGenEval", population: "AdvGenPopu") -> torch.Tensor:
        num_nets = population.config.size
        device = population.nets.config.device
        fitness_G = torch.zeros(num_nets, device=device)
        fitness_D = torch.zeros(num_nets, device=device)

        # Random shuffle pairing: network i generates, network perm[i] discriminates
        perm = torch.randperm(num_nets, device=device)
        d_indices = perm

        # Phase 1: Run all generators in parallel, collect trajectories
        population.nets.reset()
        x = self.env.reset()
        gen_trajectories: list[torch.Tensor] = []
        env_done = torch.zeros(num_nets, dtype=torch.bool, device=device)

        for step in range(self.config.max_steps):
            obs = x["observation"].to(device)
            gen_trajectories.append(obs.clone())
            x["action"] = population.get_actions(obs)
            x = self.env.step(x)["next"]
            env_done = env_done | x["done"].squeeze().to(device)

        # Phase 2: Compute D(x_G) - discriminators evaluate generator trajectories
        population.nets.reset()
        for obs in gen_trajectories:
            # Reorder obs so discriminator d_indices[i] sees generator i's obs
            # We need: for each discriminator j, it sees obs from generator inverse_perm[j]
            obs_for_disc = obs[torch.argsort(d_indices)]
            disc_scores = population.get_discrimination(obs_for_disc)
            # Generator i's fitness += discriminator's score for its trajectory
            fitness_G += disc_scores[d_indices]

        # Phase 3: Run target agent, all discriminators observe same trajectory
        population.nets.reset()
        target_env = GymEnv(self.config.env_name)
        obs_T = target_env.reset()["observation"]
        steps_T = 0

        for step in range(self.config.max_steps):
            # Get action from target agent
            obs_np = obs_T.numpy()
            action_T, _ = self.target_agent.predict(obs_np, deterministic=True)
            result = target_env.step(torch.tensor(action_T))
            obs_T = result["next"]["observation"]
            done_T = result["next"]["done"]

            # Broadcast obs_T to all discriminators
            obs_T_batch = obs_T.to(device).unsqueeze(0).expand(num_nets, -1)
            disc_scores_T = population.get_discrimination(obs_T_batch)
            fitness_D += disc_scores_T
            steps_T += 1

            if done_T:
                break

        # Normalize by trajectory lengths
        fitness_G /= len(gen_trajectories)
        fitness_D /= max(steps_T, 1)

        # Discriminator fitness = D(x_T) - D(x_G)
        # fitness_D currently = avg D(x_T), fitness_G = avg D(x_G)
        # But fitness_G is indexed by generator, need D(x_G) indexed by discriminator
        # D(x_G) for discriminator j = fitness_G[inverse_perm[j]] = fitness_G[argsort(d_indices)[j]]
        D_x_G_for_disc = fitness_G[torch.argsort(d_indices)]
        fitness_D = fitness_D - D_x_G_for_disc

        # Combined fitness: each network's total = generator fitness + discriminator fitness
        return fitness_G + fitness_D
