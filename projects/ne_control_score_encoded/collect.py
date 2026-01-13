"""Collect states from a Gym environment using a random policy.

Usage:
    python -m projects.ne_control_score_encoded.collect --env_name CartPole-v1 --num_episodes 100
"""

import argparse
import json
import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

log = logging.getLogger(__name__)


def collect_states(
    env_name: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> np.ndarray:
    """Collect states by running a random policy in the environment."""
    env = gym.make(env_name)
    all_states: list[np.ndarray] = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        all_states.append(obs)

        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            all_states.append(obs)

            if terminated or truncated:
                break

    env.close()
    return np.array(all_states, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect states from Gym environment")
    parser.add_argument("--env_name", type=str, required=True, help="Gym environment name")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info(f"Collecting states from {args.env_name}")

    states = collect_states(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    log.info(f"Collected {len(states)} states with shape {states.shape}")

    # Save to project data directory
    data_dir = Path(os.environ.get("AI_RESEARCH_PATH", ".")) / "projects" / "ne_control_score_encoded" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Clean env name for filename (e.g., CartPole-v1 -> cartpole_v1)
    clean_name = args.env_name.lower().replace("-", "_")
    output_path = data_dir / f"states_{clean_name}.npy"

    np.save(output_path, states)
    log.info(f"Saved states to {output_path}")

    # Also save metadata
    metadata = {
        "env_name": args.env_name,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "num_states": len(states),
        "state_dim": int(states.shape[1]),
    }
    metadata_path = data_dir / f"states_{clean_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
