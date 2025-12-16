"""Imitation learning and adversarial training for neuroevolution.

Implements GAN-style training with generator (imitator) and discriminator populations.
Based on old_imitate.py architecture.
"""

import copy

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from common.ne.config import StatePersistenceConfig


def hide_elements(
    obs: Tensor,
    hide_fn=None,
    indices: list[int] | None = None,
) -> Tensor:
    """Hide specific elements from observation.

    For imitation learning, the discriminator should learn to distinguish
    behavior without seeing certain privileged information (e.g., game score,
    internal state variables, remaining lives, etc.) that would make the task trivial.

    Args:
        obs: Environment observation [obs_size]
        hide_fn: Optional callable(obs) -> obs that hides/masks elements
        indices: Optional list of indices to zero out (if hide_fn not provided)

    Returns:
        Observation with elements hidden/masked

    Examples:
        # Custom hiding function
        hide_elements(obs, hide_fn=lambda x: torch.cat([x[:3], torch.zeros(1), x[4:]]))

        # Zero out specific indices (e.g., score at index 3, lives at index 7)
        hide_elements(obs, indices=[3, 7])

        # No hiding (returns obs unchanged)
        hide_elements(obs)
    """
    if hide_fn is not None:
        return hide_fn(obs)

    if indices is not None:
        obs_copy = obs.clone()
        obs_copy[indices] = 0.0
        return obs_copy

    # Default: no hiding (return unchanged)
    return obs


def evaluate_imitation_episode(
    generator_pop,
    discriminator_pop,
    target_agent,
    env,
    max_steps: int,
    hide_fn=None,
    indices: list[int] | None = None,
    state_config: StatePersistenceConfig | None = None,
    curr_gen: int = 0,
    merge_mode: bool = False,
) -> tuple[Float[Tensor, "num_gen_nets"], Float[Tensor, "num_disc_nets"]]:
    """Evaluate generator and discriminator populations on imitation task.

    Implements two-phase evaluation:
    1. Generator takes actions → Discriminator scores them
    2. Target takes actions → Discriminator scores them

    Generator fitness = p_target from phase 1 - p_target from phase 2
    Discriminator fitness = -(p_target from phase 1) + (p_target from phase 2)

    Args:
        generator_pop: Generator population (imitator)
        discriminator_pop: Discriminator population (discriminates real from fake)
        target_agent: Target agent to imitate (callable that takes obs and returns action)
        env: TorchRL environment
        max_steps: Maximum steps per evaluation
        hide_fn: Optional callable(obs) -> obs to hide elements from discriminator
        hide_indices: Optional list of indices to zero out (if hide_fn not provided)
        state_config: Optional state persistence configuration
        curr_gen: Current generation number
        merge_mode: If True, scale generator fitness as (fitness * 2 - 1)

    Returns:
        Tuple of (generator_fitness, discriminator_fitness) tensors
    """
    if state_config is None:
        state_config = StatePersistenceConfig()

    gen_nets = generator_pop.nets
    disc_nets = discriminator_pop.nets
    num_gen = gen_nets.num_nets
    num_disc = disc_nets.num_nets
    device = gen_nets.device

    # Check if networks are recurrent
    gen_is_recurrent = hasattr(gen_nets, "forward_batch_sequence")
    disc_is_recurrent = hasattr(disc_nets, "forward_batch_sequence")

    # Initialize fitness accumulators
    generator_fitness = torch.zeros(num_gen, device=device)
    discriminator_fitness = torch.zeros(num_disc, device=device)

    # === PHASE 1: Generator takes actions, Discriminator scores ===

    # Reset environment
    if hasattr(env, "set_seed"):
        env.set_seed(seed=curr_gen)
    tensordict = env.reset()
    obs = tensordict["observation"]

    # Reset hidden states
    if gen_is_recurrent:
        gen_nets.reset_hidden_states()
    if disc_is_recurrent:
        disc_nets.reset_hidden_states()

    p_target_gen = torch.zeros(num_disc, device=device)  # Discriminator's score for generator
    num_steps_gen = 0

    with torch.no_grad():
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Generator forward pass
            obs_batch = obs.unsqueeze(0).unsqueeze(0).expand(num_gen, 1, -1)

            if gen_is_recurrent:
                if hasattr(gen_nets, "forward_batch_step"):
                    if hasattr(gen_nets, "_current_hidden_states") and gen_nets._current_hidden_states:
                        h_states = [h.unsqueeze(1) for h in gen_nets._current_hidden_states]
                    else:
                        h_states = [
                            torch.zeros(num_gen, 1, gen_nets.dimensions[i + 1], device=device)
                            for i in range(gen_nets.num_layers)
                        ]
                    logits, h_new = gen_nets.forward_batch_step(obs_batch, h_states)
                    gen_nets._current_hidden_states = [h.squeeze(1) for h in h_new]
                else:
                    h_0 = gen_nets._current_hidden_states if hasattr(gen_nets, "_current_hidden_states") else None
                    logits, h_final = gen_nets.forward_batch_sequence(obs.unsqueeze(0), h_0=h_0)
                    gen_nets._current_hidden_states = h_final
                    logits = logits[:, -1:, :]
            else:
                logits = gen_nets.forward_batch(obs)
                logits = logits.unsqueeze(1)

            # Sample action from first generator network
            action_probs = F.softmax(logits, dim=-1)
            action = action_probs[0].argmax(dim=-1).squeeze().item()

            # Step environment
            tensordict = env.step(tensordict.set("action", torch.tensor(action)))
            obs = tensordict["observation"]
            done = tensordict.get("done", torch.tensor(False)).item()

            # Discriminator forward pass (hide score from observation)
            obs_hidden = hide_elements(obs, hide_fn=hide_fn, indices=hide_indices)
            obs_disc_batch = obs_hidden.unsqueeze(0).unsqueeze(0).expand(num_disc, 1, -1)

            if disc_is_recurrent:
                if hasattr(disc_nets, "forward_batch_step"):
                    if hasattr(disc_nets, "_current_hidden_states") and disc_nets._current_hidden_states:
                        h_states = [h.unsqueeze(1) for h in disc_nets._current_hidden_states]
                    else:
                        h_states = [
                            torch.zeros(num_disc, 1, disc_nets.dimensions[i + 1], device=device)
                            for i in range(disc_nets.num_layers)
                        ]
                    disc_logits, h_new = disc_nets.forward_batch_step(obs_disc_batch, h_states)
                    disc_nets._current_hidden_states = [h.squeeze(1) for h in h_new]
                else:
                    h_0 = disc_nets._current_hidden_states if hasattr(disc_nets, "_current_hidden_states") else None
                    disc_logits, h_final = disc_nets.forward_batch_sequence(obs_hidden.unsqueeze(0), h_0=h_0)
                    disc_nets._current_hidden_states = h_final
                    disc_logits = disc_logits[:, -1:, :]
            else:
                disc_logits = disc_nets.forward_batch(obs_hidden)
                disc_logits = disc_logits.unsqueeze(1)

            # Discriminator outputs probability that this is from target
            # Assuming output is [num_disc, 1, 1] - single probability value
            p_target_gen += disc_logits.squeeze()
            num_steps_gen += 1
            steps += 1

    # Average discriminator scores
    if num_steps_gen > 0:
        p_target_gen /= num_steps_gen

    # === PHASE 2: Target takes actions, Discriminator scores ===

    # Reset environment
    if hasattr(env, "set_seed"):
        env.set_seed(seed=curr_gen)
    tensordict = env.reset()
    obs = tensordict["observation"]

    # Reset discriminator hidden states
    if disc_is_recurrent:
        disc_nets.reset_hidden_states()

    p_target_target = torch.zeros(num_disc, device=device)  # Discriminator's score for target
    num_steps_target = 0

    with torch.no_grad():
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Target forward pass (target sees full observation)
            action = target_agent(obs)
            if isinstance(action, Tensor):
                action = action.item()

            # Step environment
            tensordict = env.step(tensordict.set("action", torch.tensor(action)))
            obs = tensordict["observation"]
            done = tensordict.get("done", torch.tensor(False)).item()

            # Discriminator forward pass (hide score from observation)
            obs_hidden = hide_elements(obs, hide_fn=hide_fn, indices=hide_indices)
            obs_disc_batch = obs_hidden.unsqueeze(0).unsqueeze(0).expand(num_disc, 1, -1)

            if disc_is_recurrent:
                if hasattr(disc_nets, "forward_batch_step"):
                    if hasattr(disc_nets, "_current_hidden_states") and disc_nets._current_hidden_states:
                        h_states = [h.unsqueeze(1) for h in disc_nets._current_hidden_states]
                    else:
                        h_states = [
                            torch.zeros(num_disc, 1, disc_nets.dimensions[i + 1], device=device)
                            for i in range(disc_nets.num_layers)
                        ]
                    disc_logits, h_new = disc_nets.forward_batch_step(obs_disc_batch, h_states)
                    disc_nets._current_hidden_states = [h.squeeze(1) for h in h_new]
                else:
                    h_0 = disc_nets._current_hidden_states if hasattr(disc_nets, "_current_hidden_states") else None
                    disc_logits, h_final = disc_nets.forward_batch_sequence(obs_hidden.unsqueeze(0), h_0=h_0)
                    disc_nets._current_hidden_states = h_final
                    disc_logits = disc_logits[:, -1:, :]
            else:
                disc_logits = disc_nets.forward_batch(obs_hidden)
                disc_logits = disc_logits.unsqueeze(1)

            p_target_target += disc_logits.squeeze()
            num_steps_target += 1
            steps += 1

    # Average discriminator scores
    if num_steps_target > 0:
        p_target_target /= num_steps_target

    # === Compute fitness ===

    # Generator wants discriminator to think it's the target (maximize p_target)
    # We use first discriminator's score for generator fitness
    gen_fitness_value = p_target_gen[0].item()

    if merge_mode:
        # Merge mode: scale to [-1, 1] range (fair scaling)
        gen_fitness_value = gen_fitness_value * 2 - 1

    generator_fitness[:] = gen_fitness_value

    # Discriminator wants to distinguish: high score for target, low for generator
    # Discriminator fitness = -score_for_gen + score_for_target
    discriminator_fitness = -p_target_gen + p_target_target

    # Handle continual fitness if enabled
    if state_config.fit_transfer:
        generator_pop.continual_fitness += generator_fitness
        discriminator_pop.continual_fitness += discriminator_fitness
        return -generator_pop.continual_fitness, -discriminator_pop.continual_fitness

    return -generator_fitness, -discriminator_fitness


def create_imitation_fitness_evaluators(
    generator_pop,
    discriminator_pop,
    target_agent,
    env,
    max_steps: int,
    hide_fn=None,
    indices: list[int] | None = None,
    state_config=None,
    merge_mode: bool = False,
):
    """Create fitness evaluators for imitation learning.

    Returns two closures - one for generator, one for discriminator.

    Args:
        generator_pop: Generator population
        discriminator_pop: Discriminator population
        target_agent: Target agent to imitate
        env: TorchRL environment
        max_steps: Maximum steps per evaluation
        hide_fn: Optional callable(obs) -> obs to hide elements from discriminator
        hide_indices: Optional list of indices to zero out (if hide_fn not provided)
        state_config: Optional state persistence configuration
        merge_mode: If True, scale generator fitness

    Returns:
        Tuple of (gen_fitness_fn, disc_fitness_fn)
    """
    def gen_fitness_fn() -> Float[Tensor, "num_gen_nets"]:
        """Evaluate generator fitness."""
        gen_fit, _ = evaluate_imitation_episode(
            generator_pop, discriminator_pop, target_agent, env,
            max_steps, hide_fn, indices,
            state_config, generator_pop.curr_gen, merge_mode
        )
        return gen_fit

    def disc_fitness_fn() -> Float[Tensor, "num_disc_nets"]:
        """Evaluate discriminator fitness."""
        _, disc_fit = evaluate_imitation_episode(
            generator_pop, discriminator_pop, target_agent, env,
            max_steps, hide_fn, indices,
            state_config, discriminator_pop.curr_gen, merge_mode
        )
        return disc_fit

    return gen_fitness_fn, disc_fitness_fn


def train_imitation(
    generator_pop,
    discriminator_pop,
    target_agent,
    train_env,
    test_env,
    max_steps: int,
    hide_fn=None,
    indices: list[int] | None = None,
    optimizer: str = "ga",
    max_time: int = 3600,
    eval_interval: int = 60,
    checkpoint_path_gen=None,
    checkpoint_path_disc=None,
    logger=None,
    state_config=None,
    merge_mode: bool = False,
) -> dict:
    """High-level imitation learning training.

    Orchestrates GAN-style training with generator and discriminator populations.

    Args:
        generator_pop: Generator population (imitator)
        discriminator_pop: Discriminator population (discriminates real from fake)
        target_agent: Target agent to imitate
        train_env: Training environment
        test_env: Test environment
        max_steps: Maximum steps per evaluation
        hide_fn: Optional callable(obs) -> obs to hide elements from discriminator
        hide_indices: Optional list of indices to zero out (if hide_fn not provided)
        optimizer: Which algorithm to use - "ga", "es", or "cmaes"
        max_time: Maximum training time in seconds
        eval_interval: Seconds between test evaluations
        checkpoint_path_gen: Checkpoint path for generator
        checkpoint_path_disc: Checkpoint path for discriminator
        logger: Optional logger
        state_config: Optional state persistence configuration
        merge_mode: If True, scale generator fitness as (fitness * 2 - 1)

    Returns:
        Dict with results for both populations:
            {
                "generator": {...},  # Training results for generator
                "discriminator": {...},  # Training results for discriminator
            }
    """
    # Create fitness evaluators
    gen_train_fn, disc_train_fn = create_imitation_fitness_evaluators(
        generator_pop, discriminator_pop, target_agent, train_env,
        max_steps, hide_fn, indices, state_config, merge_mode
    )
    gen_test_fn, disc_test_fn = create_imitation_fitness_evaluators(
        generator_pop, discriminator_pop, target_agent, test_env,
        max_steps, hide_fn, indices, state_config, merge_mode
    )

    # Select optimizer
    if optimizer == "ga":
        from ne.optim.ga import optimize_ga
        optimize_fn = optimize_ga
    elif optimizer == "es":
        from ne.optim.es import optimize_es
        optimize_fn = optimize_es
    elif optimizer == "cmaes":
        from ne.optim.cmaes import optimize_cmaes
        optimize_fn = optimize_cmaes
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Must be 'ga', 'es', or 'cmaes'")

    # Train both populations concurrently
    # In practice, you might want to alternate or interleave training
    # For now, we'll train them in sequence for simplicity

    # Train generator
    print("Training generator...")
    gen_results = optimize_fn(
        population=generator_pop,
        fitness_fn=gen_train_fn,
        test_fitness_fn=gen_test_fn,
        max_time=max_time // 2,  # Split time between populations
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path_gen,
        logger=logger,
        state_config=state_config,
    )

    # Train discriminator
    print("Training discriminator...")
    disc_results = optimize_fn(
        population=discriminator_pop,
        fitness_fn=disc_train_fn,
        test_fitness_fn=disc_test_fn,
        max_time=max_time // 2,
        eval_interval=eval_interval,
        checkpoint_path=checkpoint_path_disc,
        logger=logger,
        state_config=state_config,
    )

    return {
        "generator": gen_results,
        "discriminator": disc_results,
    }

