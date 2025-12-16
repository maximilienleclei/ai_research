"""Configuration classes for neuroevolution tasks."""

from dataclasses import dataclass, field
from typing import Any, Annotated as An

from hydra_zen import make_config

from common.config import BaseSubtaskConfig
from common.utils.beartype import ge, one_of
from common.utils.hydra_zen import generate_config


@dataclass
class StatePersistenceConfig:
    """Configuration for state persistence and transfer modes in continual learning.

    Combines recurrent network hidden state persistence with environment/fitness transfer modes.

    Hidden State Persistence (for recurrent networks):
        persist_across_generations: Save/restore hidden states between optimization steps
        persist_across_episodes: Maintain hidden state across episode boundaries during eval
        reset_on_selection: Reset hidden states after selection (before mutation)

    Transfer Modes (inspired by old gen_transfer.py):
        env_transfer: Save/restore environment state across generations
            - When True: Each generation continues from where previous generation left off
            - Environment state and observation are saved/restored
            - Episodes can span multiple generations
        mem_transfer: Keep agent memory (hidden states) between episodes
            - When True: Don't reset agent hidden states when episode ends
            - Only applies within a single evaluation
        fit_transfer: Accumulate fitness across all generations (continual fitness)
            - When True: Optimize continual_fitness instead of per-generation fitness
            - Fitness accumulates across entire evolutionary run

    Reset Behavior:
        The three transfer modes combine to control when agent/environment reset:
        - env_transfer=False: Reset environment each generation
        - mem_transfer=False: Reset agent when episode ends (done=True)
        - fit_transfer=True: Use continual_fitness as optimization target

    Args:
        enabled: Master switch for state persistence (ignored if transfer modes are set)
        persist_across_generations: Save/restore hidden states between optimization steps
        persist_across_episodes: Maintain hidden state across episode boundaries
        reset_on_selection: Reset hidden states after selection
        env_transfer: Save/restore environment state across generations
        mem_transfer: Keep agent memory between episodes
        fit_transfer: Accumulate fitness across all generations
    """

    enabled: bool = False
    persist_across_generations: bool = False
    persist_across_episodes: bool = False
    reset_on_selection: bool = True

    # Transfer modes from old gen_transfer.py
    env_transfer: bool = False
    mem_transfer: bool = False
    fit_transfer: bool = False


@dataclass
class NESubtaskConfig(BaseSubtaskConfig):
    """Neuroevolution subtask configuration."""

    # Optimizer settings
    optimizer: An[str, one_of("ga", "es", "cmaes")] = "ga"
    max_time: An[int, ge(1)] = 3600  # Maximum optimization time in seconds
    eval_interval: An[int, ge(1)] = 60  # Seconds between test evaluations

    # Population settings
    num_nets: An[int, ge(1)] = 100

    # Checkpoint settings
    checkpoint_path: str | None = None

    # State persistence
    state_config: StatePersistenceConfig = field(default_factory=StatePersistenceConfig)


@dataclass
class NETaskConfig(
    make_config(
        population=...,  # Will be configured in store
        train_data=...,  # Varies by eval_mode
        test_data=...,
        logger=...,
        config=generate_config(NESubtaskConfig),
    ),
):
    """Neuroevolution task configuration with Hydra defaults."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"population/network": "feedforward"},
            {"population/optimizer": "ga"},
            {"logger": "wandb"},
            "project",
            "task",
            {"task": None},
        ],
    )
