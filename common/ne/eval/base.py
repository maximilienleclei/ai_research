"""Generic evaluation functions for batched network populations.

Network-agnostic evaluation utilities that work with any batched network type.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def evaluate_feedforward(
    nets,
    observations: Float[Tensor, "N input_size"],
    actions: Int[Tensor, "N"],
) -> Float[Tensor, "num_nets"]:
    """Evaluate feedforward networks on data using cross-entropy loss.

    Args:
        nets: Batched feedforward network with forward_batch() method
        observations: Observations [N, input_size]
        actions: Ground truth actions [N]

    Returns:
        Fitness (mean cross-entropy) for each network [num_nets]
    """
    with torch.no_grad():
        # Get logits for all networks: [num_nets, N, output_size]
        logits = nets.forward_batch(observations)

        num_nets = logits.shape[0]
        N = observations.shape[0]
        output_size = logits.shape[2]

        # Expand actions to [num_nets, N]
        actions_expanded = actions.unsqueeze(0).expand(num_nets, -1)

        # Flatten for cross_entropy: [num_nets * N, output_size] and [num_nets * N]
        flat_logits = logits.reshape(-1, output_size)
        flat_actions = actions_expanded.reshape(-1)

        # Compute per-sample CE then reshape and mean per network
        per_sample_ce = F.cross_entropy(flat_logits, flat_actions, reduction="none")
        per_network_ce = per_sample_ce.view(num_nets, -1)
        fitness = per_network_ce.mean(dim=1)

    return fitness


def evaluate_recurrent(
    nets,
    observations: Float[Tensor, "seq_len input_size"],
    actions: Int[Tensor, "seq_len"],
) -> Float[Tensor, "num_nets"]:
    """Evaluate recurrent networks on sequence using cross-entropy loss.

    Args:
        nets: Batched recurrent network with forward_batch_sequence() method
        observations: Sequence observations [seq_len, input_size]
        actions: Ground truth actions [seq_len]

    Returns:
        Fitness (mean cross-entropy) for each network [num_nets]
    """
    with torch.no_grad():
        # Get logits for all networks: [num_nets, seq_len, output_size]
        logits, _ = nets.forward_batch_sequence(observations, h_0=None)

        num_nets = logits.shape[0]
        output_size = logits.shape[2]

        # Expand actions to [num_nets, seq_len]
        actions_expanded = actions.unsqueeze(0).expand(num_nets, -1)

        # Flatten for cross_entropy
        flat_logits = logits.reshape(-1, output_size)
        flat_actions = actions_expanded.reshape(-1)

        # Compute per-sample CE then reshape and mean per network
        per_sample_ce = F.cross_entropy(flat_logits, flat_actions, reduction="none")
        per_network_ce = per_sample_ce.view(num_nets, -1)
        fitness = per_network_ce.mean(dim=1)

    return fitness


def evaluate_episodes(
    nets,
    episodes: list[dict],
    eval_fn,
) -> Float[Tensor, "num_nets"]:
    """Evaluate networks on multiple episodes.

    Args:
        nets: Batched network
        episodes: List of episode dicts with 'observations' and 'actions'
        eval_fn: Evaluation function for single episode (evaluate_feedforward or evaluate_recurrent)

    Returns:
        Mean fitness across episodes for each network [num_nets]
    """
    episode_fitnesses = []

    for episode in episodes:
        obs = episode["observations"]
        act = episode["actions"]
        fitness = eval_fn(nets, obs, act)
        episode_fitnesses.append(fitness)

    # Mean fitness across episodes
    mean_fitness = torch.stack(episode_fitnesses).mean(dim=0)
    return mean_fitness


def evaluate_adversarial(
    nets,
    generator_data: tuple[Float[Tensor, "N obs_size"], Int[Tensor, "N"]],
    discriminator_data: tuple[Float[Tensor, "M obs_size"], Int[Tensor, "M"]],
    action_size: int,
) -> tuple[Float[Tensor, "num_nets"], Float[Tensor, "num_nets"]]:
    """Evaluate adversarial networks with split outputs.

    Networks output combined logits where the last dimension splits into:
    - [:action_size] = action prediction logits (generator)
    - [action_size:] = real/fake discrimination logits (discriminator)

    This enables a single network to perform both tasks, useful for imitation
    learning where the network must both generate actions and discriminate
    between real and generated behavior.

    Args:
        nets: Batched network with output_size = action_size + disc_size
        generator_data: Tuple of (observations, ground_truth_actions) for action prediction
        discriminator_data: Tuple of (observations, labels) where labels are 0=fake, 1=real
        action_size: Number of action dimensions (split point in output)

    Returns:
        gen_fitness: Generator fitness (action prediction cross-entropy) [num_nets]
        disc_fitness: Discriminator fitness (binary classification cross-entropy) [num_nets]
    """
    gen_obs, gen_actions = generator_data
    disc_obs, disc_labels = discriminator_data

    with torch.no_grad():
        # Evaluate generator performance
        if hasattr(nets, "forward_batch_sequence"):
            # Recurrent network
            gen_logits, _ = nets.forward_batch_sequence(gen_obs, h_0=None)
        else:
            # Feedforward network
            gen_logits = nets.forward_batch(gen_obs)

        # Split output: [:action_size] for actions
        action_logits = gen_logits[:, :, :action_size]  # [num_nets, N, action_size]

        num_nets = gen_logits.shape[0]
        N = gen_obs.shape[0]

        # Expand actions to [num_nets, N]
        actions_expanded = gen_actions.unsqueeze(0).expand(num_nets, -1)

        # Compute generator cross-entropy loss
        flat_action_logits = action_logits.reshape(-1, action_size)
        flat_actions = actions_expanded.reshape(-1)

        per_sample_ce = F.cross_entropy(flat_action_logits, flat_actions, reduction="none")
        per_network_ce = per_sample_ce.view(num_nets, -1)
        gen_fitness = per_network_ce.mean(dim=1)

        # Evaluate discriminator performance
        if hasattr(nets, "forward_batch_sequence"):
            disc_logits, _ = nets.forward_batch_sequence(disc_obs, h_0=None)
        else:
            disc_logits = nets.forward_batch(disc_obs)

        # Split output: [action_size:] for discrimination
        # Assuming single binary discrimination output
        disc_output = disc_logits[:, :, action_size:]  # [num_nets, M, disc_size]

        M = disc_obs.shape[0]
        disc_size = disc_output.shape[2]

        # Expand labels to [num_nets, M]
        labels_expanded = disc_labels.unsqueeze(0).expand(num_nets, -1)

        # Compute discriminator cross-entropy loss
        flat_disc_logits = disc_output.reshape(-1, disc_size)
        flat_labels = labels_expanded.reshape(-1)

        per_sample_disc_ce = F.cross_entropy(flat_disc_logits, flat_labels, reduction="none")
        per_network_disc_ce = per_sample_disc_ce.view(num_nets, -1)
        disc_fitness = per_network_disc_ce.mean(dim=1)

    return gen_fitness, disc_fitness
