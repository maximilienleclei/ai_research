"""Hydra-zen configuration store for neuroevolution."""

from hydra_zen import ZenStore
from lightning.pytorch.loggers.wandb import WandbLogger

from common.ne.config import NESubtaskConfig, NETaskConfig, StatePersistenceConfig
from common.ne.net.dynamic.population import DynamicNetPopulation
from common.ne.net.feedforward import BatchedFeedforward
from common.ne.net.recurrent import BatchedRecurrent
from common.store import store_wandb_logger_configs
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    """Store all neuroevolution configurations."""
    store(NETaskConfig, name="config")
    store_network_configs(store)
    store_optimizer_configs(store)
    store_state_persistence_configs(store)
    store_wandb_logger_configs(store, clb=WandbLogger)


def store_network_configs(store: ZenStore) -> None:
    """Store network architecture configurations."""
    # Feedforward networks
    store(
        generate_config(
            BatchedFeedforward,
            dimensions=[10, 32, 32, 4],  # Example dimensions
            num_nets=100,
            sigma_init=1e-3,
            sigma_noise=1e-2,
            device="cpu",
        ),
        name="feedforward",
        group="population/network",
    )

    # Recurrent networks - reservoir mode
    store(
        generate_config(
            BatchedRecurrent,
            dimensions=[10, 32, 32, 4],
            num_nets=100,
            model_type="reservoir",
            sigma_init=1e-3,
            sigma_noise=1e-2,
            device="cpu",
        ),
        name="recurrent_reservoir",
        group="population/network",
    )

    # Recurrent networks - trainable mode
    store(
        generate_config(
            BatchedRecurrent,
            dimensions=[10, 32, 32, 4],
            num_nets=100,
            model_type="trainable",
            sigma_init=1e-3,
            sigma_noise=1e-2,
            device="cpu",
        ),
        name="recurrent_trainable",
        group="population/network",
    )

    # Dynamic topology networks
    store(
        generate_config(
            DynamicNetPopulation,
            num_inputs=10,
            num_outputs=4,
            pop_size=100,
            device="cpu",
        ),
        name="dynamic",
        group="population/network",
    )


def store_optimizer_configs(store: ZenStore) -> None:
    """Store optimizer configurations."""
    # Genetic Algorithm
    store(
        generate_config(NESubtaskConfig, optimizer="ga"),
        name="ga",
        group="population/optimizer",
    )

    # Evolution Strategy
    store(
        generate_config(NESubtaskConfig, optimizer="es"),
        name="es",
        group="population/optimizer",
    )

    # CMA-ES
    store(
        generate_config(NESubtaskConfig, optimizer="cmaes"),
        name="cmaes",
        group="population/optimizer",
    )


def store_state_persistence_configs(store: ZenStore) -> None:
    """Store state persistence configurations."""
    # Default - no persistence
    store(
        generate_config(StatePersistenceConfig),
        name="default",
        group="state_config",
    )

    # Environment transfer - episodes span generations
    store(
        generate_config(
            StatePersistenceConfig,
            env_transfer=True,
            persist_across_generations=True,
        ),
        name="env_transfer",
        group="state_config",
    )

    # Memory transfer - keep hidden states between episodes
    store(
        generate_config(
            StatePersistenceConfig,
            mem_transfer=True,
            persist_across_episodes=True,
        ),
        name="mem_transfer",
        group="state_config",
    )

    # Fitness transfer - accumulate fitness across all generations
    store(
        generate_config(
            StatePersistenceConfig,
            fit_transfer=True,
        ),
        name="fit_transfer",
        group="state_config",
    )

    # Full continual learning - all transfers enabled
    store(
        generate_config(
            StatePersistenceConfig,
            env_transfer=True,
            mem_transfer=True,
            fit_transfer=True,
            persist_across_generations=True,
            persist_across_episodes=True,
        ),
        name="full_continual",
        group="state_config",
    )
