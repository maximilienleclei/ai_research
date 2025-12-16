"""Population wrapper - bridge between networks and eval/optim layers.

Single algorithm-agnostic class that wraps network objects and provides:
- Output → action conversion (softmax, argmax, raw)
- Population-level state management
- Clean interface for eval and optim layers
- Tracking attributes for continual learning and environment transfer

Does NOT handle selection - that logic belongs in ne/optim/.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class Population:
    """Bridge between networks and eval/optim layers.

    Responsibilities:
    - Wrap network objects
    - Handle output→action conversion
    - Maintain population-level state
    - Track episode/evaluation metrics for continual learning
    - Provide clean interface for eval and optim

    Does NOT handle selection - that's in optim/
    """

    def __init__(
        self,
        nets,
        action_mode: str = "softmax",
        device: str = "cpu",
    ) -> None:
        """Initialize population wrapper.

        Args:
            nets: Network object (any type: BatchedFeedforward, BatchedRecurrent, DynamicNetPopulation)
            action_mode: How to convert outputs to actions - "softmax", "argmax", or "raw"
            device: Computation device
        """
        self._nets = nets
        self.action_mode = action_mode
        self.device = device

        # Initialize tracking attributes for continual learning
        self._init_tracking_attributes()

        # Track current generation (updated by optimizer)
        self.curr_gen = 0

    @property
    def nets(self):
        """Access underlying network object."""
        return self._nets

    @property
    def num_nets(self) -> int:
        """Number of networks in population."""
        if hasattr(self._nets, "num_nets"):
            return self._nets.num_nets
        elif hasattr(self._nets, "pop_size"):
            return self._nets.pop_size
        else:
            raise AttributeError(
                f"Network {type(self._nets).__name__} has no num_nets or pop_size attribute"
            )

    def _init_tracking_attributes(self) -> None:
        """Initialize tracking attributes for continual learning.

        Attributes:
            curr_episode_score: Score accumulated in current episode [num_nets]
            curr_episode_num_steps: Steps taken in current episode [num_nets]
            curr_eval_score: Score accumulated in current evaluation [num_nets]
            curr_eval_num_steps: Steps taken in current evaluation [num_nets]
            total_num_steps: Total steps across all evaluations [num_nets]
            continual_fitness: Accumulated fitness across all generations [num_nets]
            saved_env: Saved environment state for env_transfer
            saved_env_out: Saved environment output for env_transfer
            logged_score: Score to log (differs based on env_transfer mode)
        """
        n = self.num_nets

        # Per-episode tracking (for env_transfer mode)
        self.curr_episode_score = torch.zeros(n, device=self.device)
        self.curr_episode_num_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        # Per-evaluation tracking
        self.curr_eval_score = torch.zeros(n, device=self.device)
        self.curr_eval_num_steps = torch.zeros(n, dtype=torch.long, device=self.device)

        # Global tracking
        self.total_num_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.continual_fitness = torch.zeros(n, device=self.device)

        # Environment state storage (for env_transfer)
        self.saved_env = None
        self.saved_env_out = None

        # Logged score (what gets logged to wandb/console)
        self.logged_score = None

    def reset_episode_tracking(self) -> None:
        """Reset per-episode tracking attributes."""
        self.curr_episode_score.zero_()
        self.curr_episode_num_steps.zero_()

    def reset_eval_tracking(self) -> None:
        """Reset per-evaluation tracking attributes."""
        self.curr_eval_score.zero_()
        self.curr_eval_num_steps.zero_()

    def reset_all_tracking(self) -> None:
        """Reset all tracking attributes (including global counters)."""
        self._init_tracking_attributes()

    def get_actions(
        self, logits: Float[Tensor, "num_nets batch action_size"]
    ) -> Tensor:
        """Convert network outputs to actions.

        Args:
            logits: Network outputs [num_nets, batch, action_size]

        Returns:
            actions: Converted actions based on action_mode
                - "softmax": Probability distribution [num_nets, batch, action_size]
                - "argmax": Action indices [num_nets, batch]
                - "raw": Unmodified logits [num_nets, batch, action_size]
        """
        if self.action_mode == "softmax":
            return F.softmax(logits, dim=-1)
        elif self.action_mode == "argmax":
            return logits.argmax(dim=-1)
        elif self.action_mode == "raw":
            return logits
        else:
            raise ValueError(
                f"Unknown action_mode: {self.action_mode}. "
                f"Must be 'softmax', 'argmax', or 'raw'"
            )

    def select_networks(self, indices: Tensor) -> None:
        """Select networks by indices and duplicate to fill population.

        Used by GA which operates on network objects, not parameters.
        Uses protocol-based dispatch to handle different network types.

        Args:
            indices: Network indices to keep [num_nets] (with duplicates to fill population)
        """
        from common.ne.net.protocol import ParameterizableNetwork, StructuralNetwork

        if isinstance(self._nets, StructuralNetwork):
            # DynamicNetPopulation: needs fitness tensor for selection
            # Reconstruct fitness ordering from indices
            dummy_fitness = self._create_fitness_from_indices(indices)
            self._nets.select_and_duplicate(dummy_fitness)

        elif isinstance(self._nets, ParameterizableNetwork):
            # BatchedFeedforward or BatchedRecurrent: direct cloning
            self._nets.clone_network(indices)

        else:
            raise TypeError(
                f"Network {type(self._nets)} doesn't support selection. "
                "Network must implement either ParameterizableNetwork or StructuralNetwork protocol."
            )

    def _create_fitness_from_indices(self, indices: Tensor) -> Tensor:
        """Helper to reconstruct fitness ordering from selection indices.

        For DynamicNetPopulation which needs fitness tensor for select_and_duplicate.

        Args:
            indices: Selection indices [num_nets]

        Returns:
            Dummy fitness tensor where selected networks have best (lowest) fitness
        """
        dummy_fitness = torch.arange(
            self.num_nets, device=self.device, dtype=torch.float
        )
        # Assign best fitness to selected networks
        for i, idx in enumerate(indices[: self.num_nets // 2].unique()):
            dummy_fitness[idx] = -1000 - i
        return dummy_fitness

    def get_parameters_flat(self) -> Float[Tensor, "num_nets num_params"]:
        """Get flattened parameters for all networks.

        Used by ES/CMA-ES which operate on parameter vectors, not network structure.
        Uses protocol-based dispatch.

        Returns:
            Flat parameter tensor [num_nets, num_params]

        Raises:
            TypeError: If network doesn't support parameter flattening (e.g., DynamicNetPopulation)
        """
        from common.ne.net.protocol import ParameterizableNetwork

        if not isinstance(self._nets, ParameterizableNetwork):
            raise TypeError(
                f"{type(self._nets).__name__} doesn't support flat parameters. "
                "Use GA which operates on network structure."
            )

        return self._nets.get_parameters_flat()

    def set_parameters_flat(self, flat_params: Float[Tensor, "num_nets num_params"]) -> None:
        """Set parameters from flattened tensor.

        Used by ES/CMA-ES to set weighted-average parameters back to networks.
        Uses protocol-based dispatch.

        Args:
            flat_params: Flat parameters [num_nets, num_params]

        Raises:
            TypeError: If network doesn't support parameter setting (e.g., DynamicNetPopulation)
        """
        from common.ne.net.protocol import ParameterizableNetwork

        if not isinstance(self._nets, ParameterizableNetwork):
            raise TypeError(
                f"{type(self._nets).__name__} doesn't support flat parameters. "
                "Use GA instead."
            )

        self._nets.set_parameters_flat(flat_params)

    def mutate(self) -> None:
        """Apply mutations to all networks.

        Delegates to network's mutate() method.
        """
        self._nets.mutate()

    def get_state_dict(self) -> dict:
        """Get state for checkpointing.

        Returns:
            State dict containing network state, population settings, and tracking attributes
        """
        state = {
            "action_mode": self.action_mode,
            # Tracking attributes
            "curr_episode_score": self.curr_episode_score,
            "curr_episode_num_steps": self.curr_episode_num_steps,
            "curr_eval_score": self.curr_eval_score,
            "curr_eval_num_steps": self.curr_eval_num_steps,
            "total_num_steps": self.total_num_steps,
            "continual_fitness": self.continual_fitness,
            "logged_score": self.logged_score,
        }

        # Add network state if available
        if hasattr(self._nets, "get_state_dict"):
            state["net_state"] = self._nets.get_state_dict()

        # Add saved environment if present (use pickle instead of deepcopy)
        if self.saved_env is not None:
            import pickle
            state["saved_env"] = pickle.dumps(self.saved_env)
        if self.saved_env_out is not None:
            import pickle
            # Handle both tensor and non-tensor cases
            if isinstance(self.saved_env_out, Tensor):
                state["saved_env_out"] = self.saved_env_out.clone()
                state["saved_env_out_is_tensor"] = True
            else:
                state["saved_env_out"] = pickle.dumps(self.saved_env_out)
                state["saved_env_out_is_tensor"] = False

        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint.

        Args:
            state: State dictionary from get_state_dict()
        """
        self.action_mode = state["action_mode"]

        # Restore tracking attributes if present
        if "curr_episode_score" in state:
            self.curr_episode_score = state["curr_episode_score"]
        if "curr_episode_num_steps" in state:
            self.curr_episode_num_steps = state["curr_episode_num_steps"]
        if "curr_eval_score" in state:
            self.curr_eval_score = state["curr_eval_score"]
        if "curr_eval_num_steps" in state:
            self.curr_eval_num_steps = state["curr_eval_num_steps"]
        if "total_num_steps" in state:
            self.total_num_steps = state["total_num_steps"]
        if "continual_fitness" in state:
            self.continual_fitness = state["continual_fitness"]
        if "logged_score" in state:
            self.logged_score = state["logged_score"]

        # Restore network state if available
        if "net_state" in state and hasattr(self._nets, "load_state_dict"):
            self._nets.load_state_dict(state["net_state"])

        # Restore saved environment if present (unpickle instead of deepcopy)
        if "saved_env" in state:
            import pickle
            self.saved_env = pickle.loads(state["saved_env"])
        if "saved_env_out" in state:
            # Handle both tensor and non-tensor cases
            if state.get("saved_env_out_is_tensor", False):
                self.saved_env_out = state["saved_env_out"].clone()
            else:
                import pickle
                self.saved_env_out = pickle.loads(state["saved_env_out"])
