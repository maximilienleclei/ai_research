"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for neuroevolution.

Adapts mean and covariance of search distribution based on fitness.
Uses diagonal approximation for efficiency with neural networks.

ONLY works with tensor-based networks (feedforward/recurrent), NOT DynamicNetPopulation.
"""

import math
from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.config import StatePersistenceConfig

from .base import optimize


class CMAESState:
    """CMA-ES algorithm state (mean, covariance, evolution paths).

    Uses diagonal approximation of covariance matrix for efficiency.
    Full CMA-ES maintains O(n²) covariance; diagonal is O(n).
    """

    def __init__(
        self,
        num_params: int,
        num_nets: int,
        device: str = "cpu",
        sigma_init: float = 0.5,
    ):
        """Initialize CMA-ES state.

        Args:
            num_params: Parameter dimensionality
            num_nets: Population size (lambda)
            device: Computation device
            sigma_init: Initial step size
        """
        self.num_params: int = num_params
        self.num_nets: int = num_nets
        self.device: str = device

        # Mean vector (search center) - initialized to zero
        self.mean: Float[Tensor, "num_params"] = torch.zeros(num_params, device=device)

        # Step size (global mutation strength)
        self.sigma: float = sigma_init

        # Diagonal covariance (coordinate-wise variances)
        self.C_diag: Float[Tensor, "num_params"] = torch.ones(num_params, device=device)

        # Evolution path for covariance adaptation
        self.p_c: Float[Tensor, "num_params"] = torch.zeros(num_params, device=device)

        # Evolution path for step-size adaptation
        self.p_sigma: Float[Tensor, "num_params"] = torch.zeros(num_params, device=device)

        # Generation counter
        self.generation: int = 0

        # Learning rates (standard CMA-ES constants)
        self.c_c: float = 4.0 / (num_params + 4.0)  # Covariance path learning rate
        self.c_1: float = 2.0 / ((num_params + 1.3) ** 2 + num_nets)  # Rank-1 update
        self.c_mu: float = min(
            1 - self.c_1,
            2 * (num_nets - 2 + 1 / num_nets) / ((num_params + 2) ** 2 + num_nets),
        )  # Rank-mu update
        self.c_sigma: float = (
            2 + num_nets
        ) / (5 + num_params + num_nets)  # Step-size learning
        self.damps: float = (
            1
            + 2 * max(0, math.sqrt((num_nets - 1) / (num_params + 1)) - 1)
            + self.c_sigma
        )

        # Expected length of random vector
        self.chi_n: float = math.sqrt(num_params) * (
            1 - 1 / (4 * num_params) + 1 / (21 * num_params**2)
        )


def select_cmaes(population, fitness: Float[Tensor, "num_nets"]) -> None:
    """CMA-ES selection: adapt search distribution based on fitness.

    Updates mean and diagonal covariance to focus search on promising regions.

    Args:
        population: Population wrapper with CMA-ES state attached
        fitness: Fitness values [num_nets] (lower is better)

    Raises:
        TypeError: If network is DynamicNetPopulation
    """
    from common.ne.net.dynamic.population import DynamicNetPopulation

    # Enforce tensor networks only
    if isinstance(population.nets, DynamicNetPopulation):
        raise TypeError(
            "CMA-ES requires tensor-based networks. Use GA for DynamicNetPopulation."
        )

    # Get or initialize CMA-ES state
    if not hasattr(population, "_cmaes_state"):
        # Initialize state on first call
        params = population.get_parameters_flat()
        num_params: int = params.shape[1]
        population._cmaes_state = CMAESState(
            num_params=num_params,
            num_nets=population.num_nets,
            device=population.device,
        )
        # Store current parameters as samples
        population._cmaes_samples = params.clone()

    state: CMAESState = population._cmaes_state
    samples: Float[Tensor, "num_nets num_params"] = population._cmaes_samples

    # Sort by fitness (lower is better)
    sorted_indices: Tensor = torch.argsort(fitness)

    # Compute selection weights (recombination weights)
    mu: int = population.num_nets // 2  # Number of parents
    weights: Float[Tensor, "mu"] = torch.log(
        torch.tensor(mu + 0.5, device=state.device)
    ) - torch.log(torch.arange(1, mu + 1, device=state.device, dtype=torch.float))
    weights = weights / weights.sum()  # Normalize
    mu_eff: float = 1.0 / (weights**2).sum().item()  # Variance effective selection mass

    # Update mean (weighted average of top mu samples)
    old_mean: Float[Tensor, "num_params"] = state.mean.clone()
    selected_samples: Float[Tensor, "mu num_params"] = samples[sorted_indices[:mu]]
    state.mean = (weights.view(-1, 1) * selected_samples).sum(dim=0)

    # Compute evolution step
    step: Float[Tensor, "num_params"] = state.mean - old_mean

    # Update evolution path for covariance (p_c)
    C_sqrt_inv: Float[Tensor, "num_params"] = 1.0 / torch.sqrt(
        state.C_diag
    )  # Diagonal inverse square root
    state.p_c = (1 - state.c_c) * state.p_c + (
        math.sqrt(state.c_c * (2 - state.c_c) * mu_eff) * step / state.sigma * C_sqrt_inv
    )

    # Update evolution path for step-size (p_sigma)
    state.p_sigma = (1 - state.c_sigma) * state.p_sigma + (
        math.sqrt(state.c_sigma * (2 - state.c_sigma) * mu_eff) * step / state.sigma
    )

    # Update step-size (sigma)
    state.sigma = state.sigma * torch.exp(
        (state.c_sigma / state.damps) * (state.p_sigma.norm() / state.chi_n - 1)
    ).item()

    # Update diagonal covariance matrix
    # Rank-1 update from evolution path
    state.C_diag = (
        (1 - state.c_1 - state.c_mu) * state.C_diag + state.c_1 * state.p_c**2
    )

    # Rank-mu update from selected samples
    for i in range(mu):
        idx: int = sorted_indices[i].item()
        diff: Float[Tensor, "num_params"] = (samples[idx] - old_mean) / state.sigma
        state.C_diag = state.C_diag + state.c_mu * weights[i] * diff**2

    # Ensure covariance stays positive
    state.C_diag = torch.clamp(state.C_diag, min=1e-10)

    state.generation += 1

    # Generate new samples from updated distribution
    noise: Float[Tensor, "num_nets num_params"] = torch.randn(
        population.num_nets, state.num_params, device=state.device
    )
    # Scale by covariance diagonal and sigma
    new_samples: Float[Tensor, "num_nets num_params"] = (
        state.mean.unsqueeze(0)
        + state.sigma * noise * torch.sqrt(state.C_diag).unsqueeze(0)
    )

    # Set parameters
    population.set_parameters_flat(new_samples)
    population._cmaes_samples = new_samples


def optimize_cmaes(
    population,
    fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    test_fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
    state_config: StatePersistenceConfig | None = None,
) -> dict:
    """Optimize networks using CMA-ES.

    Args:
        population: Population wrapper (must contain ParameterizableNetwork)
        fitness_fn: Training fitness evaluator
        test_fitness_fn: Test fitness evaluator
        max_time: Maximum time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path: Checkpoint save path
        logger: Optional logger
        state_config: State persistence config

    Returns:
        Dict with fitness_history, test_loss_history, final_generation

    Raises:
        TypeError: If population contains DynamicNetPopulation
    """
    from common.ne.net.dynamic.population import DynamicNetPopulation

    if isinstance(population.nets, DynamicNetPopulation):
        raise TypeError("CMA-ES incompatible with DynamicNetPopulation. Use GA.")

    return optimize(
        population=population,
        fitness_fn=fitness_fn,
        test_fitness_fn=test_fitness_fn,
        selection_fn=select_cmaes,
        algorithm_name="cmaes",
        max_time=max_time,
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path,
        logger=logger,
        state_config=state_config,
    )
