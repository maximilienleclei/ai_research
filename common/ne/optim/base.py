"""Shared optimization loop for evolutionary algorithms.

Extracts common training loop logic used by GA, ES, and CMA-ES.
New architecture: optimizer receives fitness functions (closures), never sees data directly.
"""

import time
from pathlib import Path
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.config import StatePersistenceConfig


def optimize(
    population,
    fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    test_fitness_fn: Callable[[], Float[Tensor, "num_nets"]],
    selection_fn: Callable,
    algorithm_name: str,
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path: Path | None = None,
    logger=None,
    state_config: StatePersistenceConfig | None = None,
) -> dict:
    """Shared optimization loop for all evolutionary algorithms.

    Args:
        population: Population wrapper (bridge between nets and optim)
        fitness_fn: Returns fitness [num_nets] on training data (closure that captures data)
        test_fitness_fn: Returns fitness [num_nets] on test data (closure)
        selection_fn: Algorithm-specific selection function (population, fitness) -> None
        algorithm_name: Name of algorithm for logging ("ga", "es", "cmaes")
        max_time: Maximum optimization time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path: Path to save/load checkpoints
        logger: Optional logger for tracking
        state_config: Optional state persistence configuration for recurrent networks

    Returns:
        Dict with fitness_history, test_loss_history, final_generation

    Note:
        fitness_fn and test_fitness_fn are closures created by eval layer.
        They capture obs/actions internally - optimizer never sees raw data.
    """
    fitness_history = []
    test_loss_history = []
    start_gen = 0
    saved_hidden_states = None

    # Try to resume from checkpoint
    if checkpoint_path and checkpoint_path.exists():
        print(f"  Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        fitness_history = checkpoint["fitness_history"]
        test_loss_history = checkpoint.get("test_loss_history", [])
        start_gen = checkpoint["generation"] + 1

        # Restore population state if available
        if "pop_state" in checkpoint:
            population.load_state_dict(checkpoint["pop_state"])

        # Restore hidden states if using state persistence
        if state_config and state_config.persist_across_generations:
            saved_hidden_states = checkpoint.get("hidden_states")

        # Restore CMA-ES state if present
        if "cmaes_state" in checkpoint:
            from common.ne.optim.cmaes import CMAESState

            cmaes_dict = checkpoint["cmaes_state"]
            state_obj = CMAESState(
                num_params=cmaes_dict["num_params"],
                num_nets=cmaes_dict["num_nets"],
                device=population.device,
            )
            state_obj.mean = cmaes_dict["mean"]
            state_obj.sigma = cmaes_dict["sigma"]
            state_obj.C_diag = cmaes_dict["C_diag"]
            state_obj.p_c = cmaes_dict["p_c"]
            state_obj.p_sigma = cmaes_dict["p_sigma"]
            state_obj.generation = cmaes_dict["generation"]
            population._cmaes_state = state_obj
            population._cmaes_samples = checkpoint["cmaes_samples"]

        print(f"  Resumed at generation {start_gen}")

    generation = start_gen
    start_time = time.time()
    last_eval_time = -eval_interval

    print(
        f"  Optimizing with {algorithm_name.upper()} for {max_time}s ({max_time/60:.1f} min)..."
    )

    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            print(f"  Time limit reached ({elapsed:.1f}s)")
            break

        # Update population's current generation (used by env evaluation)
        population.curr_gen = generation

        # Restore hidden states if persisting across generations
        if (
            state_config
            and state_config.persist_across_generations
            and saved_hidden_states is not None
        ):
            if hasattr(population.nets, "restore_hidden_states"):
                population.nets.restore_hidden_states(saved_hidden_states)

        # Evaluate fitness on training data
        fitness = fitness_fn()  # Closure - no data exposure!
        best_fitness = fitness.min().item()
        fitness_history.append(best_fitness)

        # Selection (algorithm-specific)
        selection_fn(population, fitness)

        # Mutation
        population.mutate()

        # Reset hidden states after selection if configured
        if state_config and state_config.reset_on_selection:
            if hasattr(population.nets, "reset_hidden_states"):
                population.nets.reset_hidden_states()

        # Save hidden states if persisting across generations
        if state_config and state_config.persist_across_generations:
            if hasattr(population.nets, "save_hidden_states"):
                saved_hidden_states = population.nets.save_hidden_states()

        # Periodic test evaluation
        if elapsed - last_eval_time >= eval_interval:
            test_fitness = test_fitness_fn()  # Closure - no data exposure!
            test_loss = test_fitness.min().item()
            test_loss_history.append(test_loss)
            last_eval_time = elapsed

            remaining = max_time - elapsed
            print(
                f"  Gen {generation} [{elapsed:.0f}s/{max_time}s, {remaining:.0f}s left]: "
                f"Best={best_fitness:.4f}, Test={test_loss:.4f}"
            )

            if logger:
                logger.log_progress(
                    epoch=generation, best_fitness=best_fitness, test_loss=test_loss
                )

        # Checkpoint every 5 minutes
        if checkpoint_path and (elapsed % 300 < 1.0):
            save_checkpoint(
                checkpoint_path,
                generation,
                fitness_history,
                test_loss_history,
                elapsed,
                population,
                algorithm_name,
                saved_hidden_states,
            )

        generation += 1

    # Final checkpoint
    total_time = time.time() - start_time
    print(
        f"  Complete: {generation} generations in {total_time:.1f}s ({total_time/60:.1f} min)"
    )

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            generation,
            fitness_history,
            test_loss_history,
            total_time,
            population,
            algorithm_name,
            saved_hidden_states,
        )

    return {
        "fitness_history": fitness_history,
        "test_loss_history": test_loss_history,
        "final_generation": generation,
    }


def save_checkpoint(
    path: Path,
    gen: int,
    fit_hist: list,
    test_hist: list,
    time: float,
    population,
    algorithm: str,
    hidden_states=None,
) -> None:
    """Save optimization checkpoint.

    Args:
        path: Checkpoint file path
        gen: Current generation
        fit_hist: Fitness history
        test_hist: Test loss history
        time: Elapsed time
        population: Population wrapper (must implement get_state_dict())
        algorithm: Algorithm name ("ga", "es", "cmaes")
        hidden_states: Optional saved hidden states for recurrent networks
    """
    checkpoint = {
        "generation": gen,
        "fitness_history": fit_hist,
        "test_loss_history": test_hist,
        "optim_time": time,
        "algorithm": algorithm,
    }

    # Save population state
    checkpoint["pop_state"] = population.get_state_dict()

    # Save hidden states if present
    if hidden_states is not None:
        checkpoint["hidden_states"] = hidden_states

    # Save CMA-ES state if present
    if hasattr(population, "_cmaes_state"):
        cmaes_state = population._cmaes_state
        checkpoint["cmaes_state"] = {
            "mean": cmaes_state.mean,
            "sigma": cmaes_state.sigma,
            "C_diag": cmaes_state.C_diag,
            "p_c": cmaes_state.p_c,
            "p_sigma": cmaes_state.p_sigma,
            "generation": cmaes_state.generation,
            "num_params": cmaes_state.num_params,
            "num_nets": cmaes_state.num_nets,
        }
        checkpoint["cmaes_samples"] = population._cmaes_samples

    torch.save(checkpoint, path)
