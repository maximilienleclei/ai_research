"""Encoded state evaluator that wraps GymScoreEval with autoencoder encoding."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated as An

import gymnasium as gym
import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.eval.base import BaseEval
from common.ne.eval.score.base import ScoreEvalConfig
from common.ne.popu.base import BasePopu
from common.utils.beartype import ge, not_empty
from projects.ne_control_score_encoded.nnmodule import Autoencoder, AutoencoderConfig


@dataclass
class EncodedScoreEvalConfig(ScoreEvalConfig):
    autoencoder_checkpoint: An[str, not_empty()] = "???"  # Path to trained autoencoder
    latent_dim: An[int, ge(1)] = 3  # Must match autoencoder config (default: state_dim - 1)
    normalization_stats_file: str | None = None  # Optional path to normalization stats


class EncodedGymScoreEval(BaseEval):
    """Score evaluator that encodes observations through a trained autoencoder."""

    def __init__(self: "EncodedGymScoreEval", config: EncodedScoreEvalConfig) -> None:
        self.config = config
        env_name = config.env_name

        # Create vectorized environment
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

        # Load trained autoencoder and normalization stats
        self._load_autoencoder(config.autoencoder_checkpoint)
        self._load_normalization_stats(config.normalization_stats_file)

    def _load_autoencoder(self: "EncodedGymScoreEval", checkpoint_path: str) -> None:
        """Load autoencoder from checkpoint."""
        ckpt_path = Path(checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Create autoencoder with matching config
        autoencoder_config = AutoencoderConfig(
            state_dim=self._obs_shape[-1],
            latent_dim=self.config.latent_dim,
        )
        self.autoencoder = Autoencoder(autoencoder_config)

        # Load weights from checkpoint (litmodule saves nnmodule weights with prefix)
        state_dict = checkpoint["state_dict"]
        # Remove 'nnmodule.' prefix from keys
        ae_state_dict = {
            k.replace("nnmodule.", ""): v
            for k, v in state_dict.items()
            if k.startswith("nnmodule.")
        }
        self.autoencoder.load_state_dict(ae_state_dict)
        self.autoencoder.eval()

    def _load_normalization_stats(
        self: "EncodedGymScoreEval",
        stats_file: str | None,
    ) -> None:
        """Load normalization statistics from JSON file."""
        self.state_mean: Tensor | None = None
        self.state_std: Tensor | None = None

        if stats_file is None:
            return

        stats_path = Path(stats_file)
        if not stats_path.exists():
            return

        with open(stats_path) as f:
            stats = json.load(f)

        self.state_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["std"], dtype=torch.float32)

    def retrieve_num_inputs_outputs(
        self: "EncodedGymScoreEval",
    ) -> tuple[int, int]:
        """Return latent_dim as input size (not raw obs_dim)."""
        return (self.config.latent_dim, self._num_actions)

    def retrieve_input_output_specs(
        self: "EncodedGymScoreEval",
    ) -> tuple[None, gym.spaces.Space]:
        return (None, self._action_space)

    def _encode_obs(
        self: "EncodedGymScoreEval",
        obs: Float[Tensor, "NE OD"],
    ) -> Float[Tensor, "NE LD"]:
        """Encode observations through autoencoder."""
        with torch.no_grad():
            # Normalize if we have stats
            if self.state_mean is not None:
                obs = (obs - self.state_mean) / self.state_std
            return self.autoencoder.encode(obs)

    def __call__(
        self: "EncodedGymScoreEval",
        population: BasePopu,
        generation: int = 0,
    ) -> Tensor:
        num_envs = self.config.num_workers
        fitness_scores = torch.zeros(num_envs)
        env_done = torch.zeros(num_envs, dtype=torch.bool)

        population.nets.reset()

        obs, _ = self.env.reset(seed=[self.config.seed + generation] * num_envs)
        obs = torch.from_numpy(obs).float()
        obs = self._encode_obs(obs)  # Encode observations

        for _ in range(self.config.max_steps):
            actions = population(obs)
            if self._is_discrete:
                actions_np = actions.argmax(dim=-1).numpy()
            else:
                actions_np = actions.numpy()

            raw_obs, rewards, terminated, truncated, _ = self.env.step(actions_np)
            obs = torch.from_numpy(raw_obs).float()
            obs = self._encode_obs(obs)  # Encode observations

            done = terminated | truncated
            fitness_scores += torch.from_numpy(rewards).float() * (~env_done).float()
            env_done = env_done | torch.from_numpy(done)

        return fitness_scores
