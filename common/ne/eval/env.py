"""TorchRL environment evaluation for neuroevolution populations.

Episode rollouts and batch evaluation for networks interacting with environments.
Supports continual learning with environment/memory/fitness transfer modes.
"""

import copy

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from common.ne.config import StatePersistenceConfig


def evaluate_env_episodes(
    population,
    env,
    num_episodes: int,
    max_steps_per_episode: int,
    metric: str = "return",
    state_config: StatePersistenceConfig | None = None,
    curr_gen: int = 0,
) -> Float[Tensor, "num_nets"]:
    """Evaluate population on environment episodes with continual learning support.

    Supports three transfer modes from old gen_transfer.py:
    - env_transfer: Save/restore environment state across generations
    - mem_transfer: Keep agent memory (hidden states) between episodes
    - fit_transfer: Accumulate fitness across all generations

    Args:
        population: Population wrapper with tracking attributes
        env: TorchRL EnvBase environment
        num_episodes: Number of episodes to evaluate per network
        max_steps_per_episode: Maximum steps per episode
        metric: Fitness metric - "return" (episode return) or "cross_entropy" (action prediction)
        state_config: Optional state persistence configuration
        curr_gen: Current generation number (for env seeding)

    Returns:
        Fitness tensor [num_nets] - fitness for each network
            - If fit_transfer: Returns continual_fitness (accumulated across all gens)
            - Otherwise: Returns curr_eval_score for this generation
    """
    nets = population.nets
    num_nets = nets.num_nets
    device = nets.device

    # Extract transfer mode settings
    if state_config is None:
        state_config = StatePersistenceConfig()

    env_transfer = state_config.env_transfer
    mem_transfer = state_config.mem_transfer
    fit_transfer = state_config.fit_transfer

    # Initialize eval tracking
    population.reset_eval_tracking()
    population.logged_score = None

    # Check if using recurrent network
    is_recurrent = hasattr(nets, "forward_batch_sequence")

    # PRE-EVAL RESET: Handle environment and agent state based on transfer modes
    if curr_gen > 0 and env_transfer:
        # Continue from saved environment state (env_transfer mode)
        if population.saved_env is not None:
            env = copy.deepcopy(population.saved_env)
            obs = copy.deepcopy(population.saved_env_out)
        else:
            # First generation with env_transfer - seed and reset normally
            if hasattr(env, "set_seed"):
                env.set_seed(seed=curr_gen)
            tensordict = env.reset()
            obs = tensordict["observation"]
    else:
        # Standard reset: seed and reset environment
        if hasattr(env, "set_seed"):
            env.set_seed(seed=curr_gen)
        tensordict = env.reset()
        obs = tensordict["observation"]

    # Initialize hidden states for recurrent networks
    if is_recurrent and not (state_config.persist_across_generations and curr_gen > 0):
        nets.reset_hidden_states()

    with torch.no_grad():
        steps_taken = 0
        done = False

        while steps_taken < max_steps_per_episode:
            # Expand observation for all networks: [obs_size] -> [num_nets, 1, obs_size]
            obs_batch = obs.unsqueeze(0).unsqueeze(0).expand(num_nets, 1, -1)

            # Forward pass
            if is_recurrent:
                if hasattr(nets, "forward_batch_step"):
                    # Get current hidden states or initialize
                    if hasattr(nets, "_current_hidden_states") and nets._current_hidden_states is not None:
                        h_states = [h.unsqueeze(1) for h in nets._current_hidden_states]
                    else:
                        h_states = [
                            torch.zeros(num_nets, 1, nets.dimensions[i + 1], device=device)
                            for i in range(nets.num_layers)
                        ]
                    logits, h_new = nets.forward_batch_step(obs_batch, h_states)
                    nets._current_hidden_states = [h.squeeze(1) for h in h_new]
                else:
                    # Fallback to sequence forward
                    h_0 = nets._current_hidden_states if hasattr(nets, "_current_hidden_states") else None
                    logits, h_final = nets.forward_batch_sequence(obs.unsqueeze(0), h_0=h_0)
                    nets._current_hidden_states = h_final
                    logits = logits[:, -1:, :]
            else:
                # Feedforward network
                logits = nets.forward_batch(obs)
                logits = logits.unsqueeze(1)

            # Convert to actions (deterministic argmax for now)
            action_probs = F.softmax(logits, dim=-1)
            actions = action_probs.argmax(dim=-1).squeeze(1)  # [num_nets]

            # Execute action from first network
            action_scalar = actions[0].item()
            if isinstance(obs, dict):
                tensordict = env.step(tensordict.set("action", torch.tensor(action_scalar)))
            else:
                tensordict = env.step(tensordict.set("action", torch.tensor(action_scalar)))

            # Get reward and next state
            reward = tensordict["reward"].item()
            done = tensordict.get("done", torch.tensor(False)).item()
            obs = tensordict["observation"]

            # Update tracking attributes
            population.curr_eval_score += reward
            population.curr_eval_num_steps += 1
            population.total_num_steps += 1

            if env_transfer:
                population.curr_episode_score += reward
                population.curr_episode_num_steps += 1

            if fit_transfer:
                population.continual_fitness += reward

            steps_taken += 1

            # Handle episode termination (done=True)
            if done:
                # DONE RESET: Based on transfer modes
                if env_transfer:
                    # env_transfer mode: save episode score, reset episode tracking, continue
                    population.logged_score = population.curr_episode_score.clone()
                    population.reset_episode_tracking()

                    # Reset hidden states unless mem_transfer is enabled
                    if is_recurrent and not mem_transfer:
                        nets.reset_hidden_states()

                    # Reset environment and continue
                    if hasattr(env, "set_seed"):
                        env.set_seed(seed=curr_gen)
                    tensordict = env.reset()
                    obs = tensordict["observation"]
                    done = False

                else:
                    # Standard mode: episode ends when done=True
                    break

    # POST-EVAL: Save state and determine logged score
    if env_transfer:
        # Save environment state for next generation
        population.saved_env = copy.deepcopy(env)
        population.saved_env_out = copy.deepcopy(obs)
    else:
        # Standard mode: logged score is the eval score
        population.logged_score = population.curr_eval_score.clone()

    # Reset hidden states unless mem_transfer is enabled
    if is_recurrent and not mem_transfer:
        nets.reset_hidden_states()

    # Return fitness based on transfer mode
    if fit_transfer:
        # Optimize continual fitness (accumulated across all generations)
        fitness = -population.continual_fitness
    else:
        # Optimize current evaluation score
        fitness = -population.curr_eval_score

    return fitness


def evaluate_env_batch(
    nets,
    env,
    observations: Float[Tensor, "N obs_size"],
    actions: Int[Tensor, "N"] | None = None,
) -> Float[Tensor, "num_nets"]:
    """Evaluate networks on batch of states from pre-recorded episodes.

    For offline evaluation using recorded environment trajectories.

    Args:
        nets: Batched network (BatchedFeedforward/BatchedRecurrent)
        env: TorchRL environment (used for metadata, not stepped)
        observations: Batch of observations [N, obs_size]
        actions: Optional ground truth actions [N] for cross-entropy evaluation

    Returns:
        Fitness tensor [num_nets] - cross-entropy loss for each network
    """
    if actions is None:
        raise ValueError("Ground truth actions required for batch evaluation")

    with torch.no_grad():
        # Check if recurrent or feedforward
        if hasattr(nets, "forward_batch_sequence"):
            # Recurrent: treat as sequence
            logits, _ = nets.forward_batch_sequence(observations, h_0=None)
        else:
            # Feedforward: batch forward
            logits = nets.forward_batch(observations)

        # Compute cross-entropy loss
        num_nets = logits.shape[0]
        output_size = logits.shape[-1]

        # Expand actions to [num_nets, N]
        actions_expanded = actions.unsqueeze(0).expand(num_nets, -1)

        # Flatten for cross_entropy
        flat_logits = logits.reshape(-1, output_size)
        flat_actions = actions_expanded.reshape(-1)

        # Compute per-sample CE then reshape and mean per network
        per_sample_ce = F.cross_entropy(flat_logits, flat_actions, reduction="none")
        per_network_ce = per_sample_ce.view(num_nets, -1)
        fitness = per_network_ce.mean(dim=1)

    return fitness
