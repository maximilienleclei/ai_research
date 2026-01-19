"""Human Action Prediction evaluator for behavior cloning from human data.

Shapes:
    NN: num_nets
    OD: obs_dim (base observation dimension)
    ID: input_dim (obs_dim + 2 if use_cl_info, else obs_dim)
    NA: n_actions
    NS: num_steps (total steps in dataset)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from gymnasium import spaces
from jaxtyping import Float, Int
from torch import Tensor

from common.ne.eval.base import BaseEval
from common.ne.popu.nets.dynamic.base import DynamicNets

if TYPE_CHECKING:
    from common.ne.popu.actor import ActorPopu


# Environment configurations
ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {"obs_dim": 4, "n_actions": 2, "gym_name": "CartPole-v1"},
    "mountaincar": {"obs_dim": 2, "n_actions": 3, "gym_name": "MountainCar-v0"},
    "acrobot": {"obs_dim": 6, "n_actions": 3, "gym_name": "Acrobot-v1"},
    "lunarlander": {"obs_dim": 8, "n_actions": 4, "gym_name": "LunarLander-v3"},
}


def compute_session_run_ids(
    timestamps: list[str],
) -> tuple[list[int], list[int]]:
    """Compute session and run IDs from episode timestamps.

    A new session begins if >= 30 minutes have passed since the previous episode.
    Within a session, runs are numbered sequentially starting from 0.

    Args:
        timestamps: List of ISO format timestamp strings

    Returns:
        Tuple of (session_ids, run_ids) - both are lists of integers
    """
    if len(timestamps) == 0:
        return [], []

    dt_list: list[datetime] = [datetime.fromisoformat(ts) for ts in timestamps]

    session_ids: list[int] = []
    run_ids: list[int] = []

    current_session: int = 0
    current_run: int = 0

    session_ids.append(current_session)
    run_ids.append(current_run)

    # Threshold: 30 minutes = 1800 seconds
    session_threshold_seconds: float = 30 * 60

    for i in range(1, len(dt_list)):
        time_diff: float = (dt_list[i] - dt_list[i - 1]).total_seconds()

        if time_diff >= session_threshold_seconds:
            current_session += 1
            current_run = 0
        else:
            current_run += 1

        session_ids.append(current_session)
        run_ids.append(current_run)

    return session_ids, run_ids


def normalize_session_run_features(
    session_ids: list[int], run_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize session and run IDs to [-1, 1] range.

    Sessions are mapped with equal spacing across all data.
    Runs are mapped with equal spacing within each session.

    Args:
        session_ids: List of session IDs (integers)
        run_ids: List of run IDs (integers)

    Returns:
        Tuple of (normalized_sessions, normalized_runs) as numpy arrays
    """
    session_arr: np.ndarray = np.array(session_ids, dtype=np.int64)
    run_arr: np.ndarray = np.array(run_ids, dtype=np.int64)

    unique_sessions: np.ndarray = np.unique(session_arr)
    num_sessions: int = len(unique_sessions)

    if num_sessions == 1:
        normalized_sessions: np.ndarray = np.zeros(
            len(session_arr), dtype=np.float32
        )
    else:
        session_to_normalized: dict[int, float] = {
            s: -1.0 + 2.0 * i / (num_sessions - 1)
            for i, s in enumerate(unique_sessions)
        }
        normalized_sessions = np.array(
            [session_to_normalized[s] for s in session_arr], dtype=np.float32
        )

    normalized_runs: np.ndarray = np.zeros(len(run_arr), dtype=np.float32)

    for session_id in unique_sessions:
        mask: np.ndarray = session_arr == session_id
        runs_in_session: np.ndarray = run_arr[mask]
        unique_runs: np.ndarray = np.unique(runs_in_session)
        num_runs: int = len(unique_runs)

        if num_runs == 1:
            normalized_runs[mask] = 0.0
        else:
            run_to_normalized: dict[int, float] = {
                r: -1.0 + 2.0 * i / (num_runs - 1)
                for i, r in enumerate(unique_runs)
            }
            for idx in np.where(mask)[0]:
                normalized_runs[idx] = run_to_normalized[run_arr[idx]]

    return normalized_sessions, normalized_runs


@dataclass
class HumanActPredEvalConfig:
    env_name: str  # cartpole, mountaincar, acrobot, lunarlander
    subject: str = "sub01"  # sub01, sub02
    use_cl_info: bool = False  # Whether to include session/run features
    batch_size: int = 1024  # Steps per fitness evaluation batch
    data_dir: str = field(
        default_factory=lambda: os.path.join(
            os.environ.get("AI_RESEARCH_PATH", "."),
            "data",
            "human_behaviour_control_tasks",
        )
    )


class HumanActPredEval(BaseEval):
    """Human action prediction evaluator for behavior cloning.

    Loads pre-recorded human behavior data and evaluates population fitness
    based on predicting human actions from observations.

    Fitness: soft accuracy (softmax probability of correct action)
    """

    def __init__(self: "HumanActPredEval", config: HumanActPredEvalConfig) -> None:
        self.config = config

        if config.env_name not in ENV_CONFIGS:
            raise ValueError(
                f"Unknown env: {config.env_name}. "
                f"Supported: {list(ENV_CONFIGS.keys())}"
            )

        self._env_config = ENV_CONFIGS[config.env_name]
        self._obs_dim = self._env_config["obs_dim"]
        self._n_actions = self._env_config["n_actions"]
        self._input_dim = self._obs_dim + 2 if config.use_cl_info else self._obs_dim

        # Build action space
        self._action_space = spaces.Discrete(self._n_actions)
        self._obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._input_dim,), dtype=np.float32
        )

        # Load and preprocess data
        self._load_data()

    def _load_data(self: "HumanActPredEval") -> None:
        """Load human behavior data from JSON file."""
        data_file = Path(self.config.data_dir) / f"{self.config.subject}_data_{self.config.env_name}.json"

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r") as f:
            episodes: list[dict] = json.load(f)

        # Compute session and run IDs
        timestamps: list[str] = [ep["timestamp"] for ep in episodes]
        session_ids, run_ids = compute_session_run_ids(timestamps)

        # Expand to step level
        step_session_ids: list[int] = []
        step_run_ids: list[int] = []
        observations: list[list[float]] = []
        actions: list[int] = []

        for ep_idx, episode in enumerate(episodes):
            for step in episode["steps"]:
                observations.append(step["observation"])
                actions.append(step["action"])
                step_session_ids.append(session_ids[ep_idx])
                step_run_ids.append(run_ids[ep_idx])

        # Convert to numpy
        obs_np: np.ndarray = np.array(observations, dtype=np.float32)
        act_np: np.ndarray = np.array(actions, dtype=np.int64)

        # Optionally add CL features
        if self.config.use_cl_info:
            norm_sessions, norm_runs = normalize_session_run_features(
                step_session_ids, step_run_ids
            )
            cl_features = np.stack([norm_sessions, norm_runs], axis=1)
            obs_np = np.concatenate([obs_np, cl_features], axis=1)

        # Store as tensors (will be moved to device on first call)
        self._observations: Float[Tensor, "NS ID"] = torch.from_numpy(obs_np)
        self._actions: Int[Tensor, "NS"] = torch.from_numpy(act_np)
        self._num_steps = len(actions)
        self._device_set = False

        # Store metadata
        self._num_episodes = len(episodes)
        self._num_sessions = len(set(session_ids))

    def retrieve_num_inputs_outputs(self: "HumanActPredEval") -> tuple[int, int]:
        """Returns (input_dim, n_actions)."""
        return (self._input_dim, self._n_actions)

    def retrieve_action_space(self: "HumanActPredEval") -> spaces.Space:
        """Returns the gymnasium action space."""
        return self._action_space

    def retrieve_input_output_specs(
        self: "HumanActPredEval",
    ) -> tuple[spaces.Space, spaces.Space]:
        """Returns (observation_space, action_space)."""
        return (self._obs_space, self._action_space)

    def _get_action_logits(
        self: "HumanActPredEval", population: "ActorPopu", x: Tensor
    ) -> Tensor:
        """Get raw action logits from population."""
        if isinstance(population.nets, DynamicNets):
            return population.nets(x)
        else:
            x = x.unsqueeze(1)
            logits = population.nets(x)
            return logits.squeeze(1)

    def __call__(
        self: "HumanActPredEval", population: "ActorPopu", generation: int = 0
    ) -> Float[Tensor, "NN"]:
        num_nets: int = population.config.size
        device = population.nets.config.device

        # Move data to device on first call
        if not self._device_set:
            self._observations = self._observations.to(device)
            self._actions = self._actions.to(device)
            self._device_set = True

        fitness: Float[Tensor, "NN"] = torch.zeros(num_nets, device=device)

        # Reset network state
        population.nets.reset()

        # Sample batch indices (deterministic based on generation for reproducibility)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(generation)
        batch_indices = torch.randperm(self._num_steps, generator=rng)[
            : self.config.batch_size
        ]
        batch_indices = batch_indices.to(device)

        # Get batch data
        batch_obs: Float[Tensor, "BS ID"] = self._observations[batch_indices]
        batch_actions: Int[Tensor, "BS"] = self._actions[batch_indices]
        batch_size = len(batch_indices)

        # Evaluate each step in batch
        for step_idx in range(batch_size):
            obs: Float[Tensor, "NN ID"] = batch_obs[step_idx].unsqueeze(0).expand(
                num_nets, -1
            )
            target_action: int = batch_actions[step_idx].item()

            # Get population's action logits
            pred_logits = self._get_action_logits(population, obs)

            # Soft accuracy: softmax probability of correct action
            probs = torch.softmax(pred_logits, dim=-1)
            correct_probs = probs[:, target_action]
            fitness += correct_probs

        # Normalize by batch size
        fitness = fitness / batch_size

        return fitness
