"""Dynamic network evolution for topology-evolving neural networks.

This module implements a graph-based neural network that can grow and prune
nodes during evolution. Unlike fixed-architecture networks, these networks
evolve their topology alongside their weights.

Architecture Overview
---------------------
Networks consist of three types of nodes:

1. **Input nodes**: Non-parametric nodes that receive external input signals.
   One input node per observation dimension.

2. **Hidden nodes**: Parametric nodes with learnable connections. Each hidden
   node has at most MAX_INCOMING_CONNECTIONS (3) incoming connections with
   associated weights. Outputs are computed as: standardize(weights · inputs).

3. **Output nodes**: Same as hidden nodes, but their outputs are the network's
   final predictions. One output node per action dimension.

Evolution Operators
-------------------
- **grow_node**: Adds a new hidden node with random connections to nearby nodes.
- **prune_node**: Removes a hidden node and cascades to disconnect orphaned nodes.
- **mutate**: Applies parameter perturbations and architectural mutations.

Design Decisions
----------------
- MAX_INCOMING_CONNECTIONS = 3: Limits node complexity while allowing non-trivial
  computation. This is a design choice balancing expressiveness vs. search space.

- Local connectivity bias: New connections are biased toward nearby nodes in the
  graph topology, controlled by `local_connectivity_probability`. This encourages
  modular structures.

- Welford running standardization: Each node maintains running mean/variance
  statistics for input normalization, enabling stable learning despite topology
  changes.

- State dict cloning: Uses state_dict serialization instead of deepcopy to avoid
  Python's recursion limit on large circular graph structures.

Shapes
------
- NMN: Number of mutable (hidden + output) nodes
- NON: Number of output nodes
- NN: Total number of nodes (input + hidden + output)

See Also
--------
- base.py: DynamicNets class that manages a population of Net instances
- utils.py: WelfordRunningStandardizer for batched normalization
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated as An

import torch
from jaxtyping import Float, Int
from ordered_set import OrderedSet
from torch import Tensor

from common.utils.beartype import ge, le, one_of

# Maximum number of incoming connections per hidden/output node.
# This limits the complexity of each node's computation while still allowing
# meaningful transformations. Value of 3 was chosen empirically.
MAX_INCOMING_CONNECTIONS = 3


class Node:
    """A single node (neuron) in a dynamic network graph.

    Nodes are the fundamental units of computation in dynamic networks. They
    maintain connections to other nodes and compute weighted sums of inputs.

    Attributes
    ----------
    role : str
        Node type: "input", "hidden", or "output".
    mutable_uid : int
        Position-based ID that changes when nodes are added/removed.
        Used for tensor indexing during forward pass.
    immutable_uid : int
        Permanent ID assigned at creation. Never changes, used for cloning.
    out_nodes : list[Node]
        Nodes this node sends signals to.
    in_nodes : list[Node]
        Nodes this node receives signals from (hidden/output only).
    weights : list[float]
        Connection weights for incoming connections (hidden/output only).
        Length is MAX_INCOMING_CONNECTIONS (3).

    Notes
    -----
    - Input nodes have no incoming connections or weights.
    - Hidden/output nodes have at most MAX_INCOMING_CONNECTIONS inputs.
    - Weights are randomly initialized when connections are made.
    """

    def __init__(
        self: "Node",
        role: An[str, one_of("input", "hidden", "output")],
        mutable_uid: An[int, ge(0)],
        immutable_uid: An[int, ge(0)],
    ) -> None:
        """Create a new node.

        Parameters
        ----------
        role : str
            Node type: "input", "hidden", or "output".
        mutable_uid : int
            Current position in the network's node list.
        immutable_uid : int
            Permanent unique identifier.
        """
        self.role: An[str, one_of("input", "hidden", "output")] = role
        # mutable_uid changes when nodes are added/removed (for tensor indexing)
        self.mutable_uid: int = mutable_uid
        # immutable_uid is permanent (for serialization/cloning)
        self.immutable_uid: int = immutable_uid
        self.out_nodes: list[Node] = []

        if self.role != "input":
            self.in_nodes: list[Node] = []
            # Initialize weights array with zeros; actual weights set on connect
            self.weights: list[float] = [0.0] * MAX_INCOMING_CONNECTIONS

    def __repr__(self: "Node") -> str:
        """Examples:
        Input node:  ('x',) → (0, '0') → ((6, '5'), (8, '7'))
        Hidden node: ((0, '0'), (6, '5')) → (8, '7') → ((3, '3'),)
        Output node: ((7, '6'), (8, '7')) → (3, '3') → ('y',)
        """
        in_nodes: str = ""
        if self.role == "input":
            in_nodes = "x"
        else:
            for node in self.in_nodes:
                in_nodes += f"{node.mutable_uid}-{node.immutable_uid},"
            in_nodes = in_nodes[:-1]

        out_nodes: str = ""
        if self.role == "output":
            out_nodes = "y"
        else:
            for node in self.out_nodes:
                out_nodes += f"{node.mutable_uid}-{node.immutable_uid},"
            out_nodes = out_nodes[:-1]
        return (
            str([] if self.role == "input" else self.weights)
            + " : "
            + in_nodes
            + " → "
            + f"{self.mutable_uid}-{self.immutable_uid}"
            + " → "
            + out_nodes
        )

    def sample_nearby_node(
        self: "Node",
        nodes_considered: OrderedSet["Node"],
        local_connectivity_probability: float,
    ) -> "Node":
        """Sample a node biased toward graph-local neighbors.

        Implements a distance-weighted sampling where closer nodes (in graph
        distance) are more likely to be sampled. This encourages modular
        network structures.

        Parameters
        ----------
        nodes_considered : OrderedSet[Node]
            Set of candidate nodes to sample from.
        local_connectivity_probability : float
            Probability of accepting a local match at each distance level.
            Higher values bias toward closer nodes.

        Returns
        -------
        Node
            A sampled node from nodes_considered.

        Algorithm
        ---------
        1. Start with immediate neighbors (distance 1)
        2. At each distance level, accept with probability local_connectivity_probability
        3. If not accepted, expand search to distance i+1
        4. Repeat until a node is sampled or all nodes are reached
        """
        # Start with nodes within distance of 1
        nodes_within_distance_i: OrderedSet[Node] = OrderedSet(
            ([] if self.role == "input" else self.in_nodes) + self.out_nodes
        )

        node_found: bool = False
        while not node_found:
            nodes_considered_at_distance_i: OrderedSet[Node] = (
                nodes_within_distance_i & nodes_considered
            )

            # Accept with probability local_connectivity_probability, or
            # if we've exhausted all reachable nodes
            should_accept = (
                local_connectivity_probability > random.random()
                and nodes_considered_at_distance_i
            ) or nodes_within_distance_i == nodes_considered

            if should_accept:
                nearby_node: Node = random.choice(nodes_considered_at_distance_i)
                node_found = True
            else:
                # Expand search to distance i+1
                nodes_within_distance_iplus1: OrderedSet[Node] = (
                    nodes_within_distance_i.copy()
                )
                for node in nodes_within_distance_i:
                    nodes_within_distance_iplus1 |= OrderedSet(
                        ([] if node.role == "input" else node.in_nodes)
                        + node.out_nodes,
                    )

                if nodes_within_distance_iplus1 != nodes_within_distance_i:
                    nodes_within_distance_i = nodes_within_distance_iplus1
                else:
                    # Reached end of connected subgraph, expand to all candidates
                    nodes_within_distance_i = OrderedSet(nodes_considered)

        return nearby_node

    def connect_to(self: "Node", node: "Node") -> None:
        """Create a connection from self to another node.

        Adds self to node's input list and assigns a random weight.

        Parameters
        ----------
        node : Node
            Target node to connect to (must be hidden or output).

        Notes
        -----
        Weight is initialized from standard normal distribution N(0,1).
        """
        weight: float = torch.randn(1).item()
        node.weights[len(node.in_nodes)] = weight
        self.out_nodes.append(node)
        node.in_nodes.append(self)

    def disconnect_from(self: "Node", node: "Node") -> None:
        """Remove connection from self to another node.

        Updates the target node's weight array to maintain contiguity.

        Parameters
        ----------
        node : Node
            Target node to disconnect from.

        Notes
        -----
        Weights are shifted left to fill the gap, maintaining the invariant
        that weights[0:len(in_nodes)] are the active weights.
        """
        i = node.in_nodes.index(self)

        # Shift weights left to fill the gap at position i
        for j in range(i, MAX_INCOMING_CONNECTIONS - 1):
            node.weights[j] = node.weights[j + 1]
        node.weights[MAX_INCOMING_CONNECTIONS - 1] = 0.0

        self.out_nodes.remove(node)
        node.in_nodes.remove(self)


@dataclass
class NodeList:
    """Container for categorized node lists.

    Provides convenient access to nodes by their role and connection status.
    Used by Net to manage its graph structure.

    Attributes
    ----------
    all : list[Node]
        All nodes in the network.
    input : list[Node]
        Input nodes (one per observation dimension).
    hidden : list[Node]
        Hidden nodes (evolved during mutation).
    output : list[Node]
        Output nodes (one per action dimension).
    receiving : list[Node]
        Nodes with at least one incoming connection. Nodes appear once per
        incoming connection (for connection counting).
    emitting : list[Node]
        Nodes with at least one outgoing connection. Nodes appear once per
        outgoing connection (for connection counting).
    being_pruned : list[Node]
        Nodes currently being pruned. Used to prevent infinite loops during
        cascading prune operations.
    """

    all: list["Node"] = field(default_factory=list)
    input: list["Node"] = field(default_factory=list)
    hidden: list["Node"] = field(default_factory=list)
    output: list["Node"] = field(default_factory=list)
    receiving: list["Node"] = field(default_factory=list)
    emitting: list["Node"] = field(default_factory=list)
    being_pruned: list["Node"] = field(default_factory=list)

    def __iter__(
        self: "NodeList",
    ) -> Iterator[list["Node"]]:
        """Iterate over all node lists for bulk operations."""
        return iter(
            [
                self.all,
                self.input,
                self.hidden,
                self.output,
                self.receiving,
                self.emitting,
                self.being_pruned,
            ],
        )


class Net:
    """A dynamic neural network with evolving topology.

    This network can grow and prune nodes during evolution, unlike fixed-
    architecture networks. It maintains a graph of connected nodes and
    tensors for efficient batched computation.

    Attributes
    ----------
    num_inputs : int
        Number of input nodes (observation dimensions).
    num_outputs : int
        Number of output nodes (action dimensions).
    device : str
        PyTorch device for tensor operations.
    nodes : NodeList
        Container for all nodes organized by role.
    weights_list : list[list[float]]
        Weights for each mutable node. Kept as Python lists for mutation,
        converted to tensors for computation.
    n_mean_m2_x_z : Tensor
        Welford running statistics and outputs for each node.
        Columns: [n, mean, m2, x, z] where:
        - n: count of observations
        - mean: running mean
        - m2: running sum of squared deviations (for variance)
        - x: raw node output
        - z: standardized output = (x - mean) / std

    Evolvable Parameters
    --------------------
    avg_num_grow_mutations : float
        Expected number of grow_node calls per mutate(). Self-adapts.
    avg_num_prune_mutations : float
        Expected number of prune_node calls per mutate(). Self-adapts.
    num_network_passes_per_input : int
        Number of forward passes per input (for recurrent-like behavior).
    local_connectivity_probability : float
        Bias toward local connections (higher = more modular structures).
    """

    def __init__(
        self: "Net",
        num_inputs: An[int, ge(1)],
        num_outputs: An[int, ge(1)],
        device: str = "cpu",
    ) -> None:
        """Create a new dynamic network.

        Parameters
        ----------
        num_inputs : int
            Number of observation dimensions.
        num_outputs : int
            Number of action dimensions.
        device : str, optional
            PyTorch device (default: "cpu").
        """
        self.num_inputs: An[int, ge(1)] = num_inputs
        self.num_outputs: An[int, ge(1)] = num_outputs
        self.device: str = device
        self.total_num_nodes_grown: An[int, ge(0)] = 0
        self.nodes: NodeList = NodeList()

        # Weights for mutable nodes (output + hidden)
        self.weights_list: list[list[float]] = []

        # Welford statistics tensor: [n, mean, m2, x, z] per node
        # - n, mean, m2: running standardization stats
        # - x: raw output, z: standardized output
        self.n_mean_m2_x_z: Float[Tensor, "NN 5"] = torch.zeros(
            (0, 5), device=self.device
        )

        # Self-adapting mutation parameters
        self.avg_num_grow_mutations: An[float, ge(0)] = 1.0
        self.avg_num_prune_mutations: An[float, ge(0)] = 0.5
        self.num_network_passes_per_input: An[int, ge(1)] = 1
        self.local_connectivity_probability: An[float, ge(0), le(1)] = 0.5

        self.initialize_architecture()

    def initialize_architecture(self: "Net") -> None:
        for _ in range(self.num_inputs):
            self.grow_node(role="input")
        for _ in range(self.num_outputs):
            self.grow_node(role="output")

    def grow_node(
        self: "Net",
        in_node_1: Node | None = None,
        role: An[str, one_of("input", "hidden", "output")] = "hidden",
    ) -> Node:
        """Method first called during initialization to grow the irremovable
        input and output nodes.

        Post-initialization, all calls create new hidden nodes.
        In such setting, three existing nodes are sampled: 2 to connect from
        and 1 to connect to."""
        new_node = Node(
            role,
            mutable_uid=len(self.nodes.all),
            immutable_uid=self.total_num_nodes_grown,
        )
        self.nodes.all.append(new_node)
        if role == "input":
            self.nodes.input.append(new_node)
            self.nodes.receiving.append(new_node)
        elif role == "output":
            self.nodes.output.append(new_node)
        else:  # role == "hidden"
            receiving_nodes_set: OrderedSet[Node] = OrderedSet(
                self.nodes.receiving
            )
            non_emitting_input_nodes: OrderedSet[Node] = OrderedSet(
                self.nodes.input
            ) - (
                OrderedSet(self.nodes.input) & OrderedSet(self.nodes.emitting)
            )
            non_receiving_output_nodes: OrderedSet[Node] = OrderedSet(
                self.nodes.output
            ) - (
                OrderedSet(self.nodes.output)
                & OrderedSet(self.nodes.receiving)
            )
            # 1) `in_node_1' → `new_node`
            if not in_node_1:
                # First focus on connecting input nodes to the rest of the
                # network.
                nodes_considered_for_in_node_1: OrderedSet[Node] = (
                    non_emitting_input_nodes
                    if non_emitting_input_nodes
                    else receiving_nodes_set
                )
                in_node_1 = random.choice(nodes_considered_for_in_node_1)
            non_emitting_input_nodes -= OrderedSet([in_node_1])
            self.grow_connection(in_node=in_node_1, out_node=new_node)
            # 2) `in_node_2' → `new_node`
            nodes_considered_for_in_node_2: OrderedSet[Node] = (
                non_emitting_input_nodes
                if non_emitting_input_nodes
                else receiving_nodes_set
            ) - OrderedSet([in_node_1])
            in_node_2: Node = in_node_1.sample_nearby_node(
                nodes_considered_for_in_node_2,
                self.local_connectivity_probability,
            )
            self.grow_connection(in_node=in_node_2, out_node=new_node)
            # 3) `new_node' → `out_node_1`
            if non_receiving_output_nodes:
                # First focus on connecting output nodes to the rest of the
                # network.
                nodes_considered_for_out_node_1: OrderedSet[Node] = (
                    non_receiving_output_nodes.copy()
                )
            else:
                nodes_considered_for_out_node_1: OrderedSet[Node] = (
                    OrderedSet()
                )
                for node in self.nodes.hidden + self.nodes.output:
                    if len(node.in_nodes) < MAX_INCOMING_CONNECTIONS:
                        nodes_considered_for_out_node_1.add(node)
            out_node_1: Node = in_node_2.sample_nearby_node(
                nodes_considered_for_out_node_1,
                self.local_connectivity_probability,
            )
            self.grow_connection(in_node=new_node, out_node=out_node_1)
            self.nodes.hidden.append(new_node)
        if role in ["hidden", "output"]:
            self.weights_list.append(new_node.weights)
        self.n_mean_m2_x_z = torch.cat(
            (self.n_mean_m2_x_z, torch.zeros((1, 5), device=self.device))
        )
        self.total_num_nodes_grown += 1
        return new_node

    def grow_connection(self: "Net", in_node: Node, out_node: Node) -> None:
        in_node.connect_to(out_node)
        self.nodes.receiving.append(out_node)
        self.nodes.emitting.append(in_node)

    def prune_node(self: "Net", node_being_pruned: Node | None = None) -> None:
        """Removes an existing hidden node."""
        if not node_being_pruned:
            if len(self.nodes.hidden) == 0:
                return
            node_being_pruned = random.choice(self.nodes.hidden)
        if node_being_pruned in self.nodes.being_pruned:
            return
        self.nodes.being_pruned.append(node_being_pruned)
        # Use identity comparison (is) instead of equality (==) to remove
        # the exact weights object, not just a matching value. Two nodes
        # can have the same weight values, so equality-based remove() would
        # remove the wrong entry.
        for i, w in enumerate(self.weights_list):
            if w is node_being_pruned.weights:
                del self.weights_list[i]
                break
        pruned_uid: int = node_being_pruned.mutable_uid
        self.n_mean_m2_x_z = torch.cat(
            (
                self.n_mean_m2_x_z[:pruned_uid],
                self.n_mean_m2_x_z[pruned_uid + 1 :],
            )
        )
        # Decrement mutable_uid BEFORE recursive pruning to keep indices valid
        for node in self.nodes.all:
            if node.mutable_uid > pruned_uid:
                node.mutable_uid -= 1
        for node_being_pruned_out_node in node_being_pruned.out_nodes.copy():
            self.prune_connection(
                in_node=node_being_pruned,
                out_node=node_being_pruned_out_node,
                node_being_pruned=node_being_pruned,
            )
        for node_being_pruned_in_node in node_being_pruned.in_nodes.copy():
            self.prune_connection(
                in_node=node_being_pruned_in_node,
                out_node=node_being_pruned,
                node_being_pruned=node_being_pruned,
            )
        for node_list in self.nodes:
            while node_being_pruned in node_list:
                node_list.remove(node_being_pruned)

    def prune_connection(
        self: "Net", in_node: Node, out_node: Node, node_being_pruned: Node
    ) -> None:
        """Called by `prune_node` to remove the `node_being_pruned`'s
        connections.

        Any hidden node that becomes disconnected from the network as a result
        is also pruned."""
        if in_node not in out_node.in_nodes:
            return
        in_node.disconnect_from(out_node)
        self.nodes.receiving.remove(out_node)
        self.nodes.emitting.remove(in_node)
        if (
            in_node is not node_being_pruned
            and in_node in self.nodes.hidden
            and in_node not in self.nodes.emitting
        ):
            self.prune_node(in_node)
        if (
            out_node is not node_being_pruned
            and out_node in self.nodes.hidden
            and out_node not in self.nodes.receiving
        ):
            self.prune_node(out_node)

    def mutate(self: "Net") -> None:
        # PARAMETER PERTURBATION
        # `avg_num_grow_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.avg_num_grow_mutations *= rand_val
        # `avg_num_prune_mutations`
        rand_val: float = 1.0 + 0.01 * torch.randn(1).item()
        self.avg_num_prune_mutations *= rand_val
        # `num_network_passes_per_input`
        rand_val: An[int, ge(1), le(100)] = torch.randint(1, 101, (1,)).item()
        if rand_val == 1 and self.num_network_passes_per_input != 1:
            self.num_network_passes_per_input -= 1
        if rand_val == 100:
            self.num_network_passes_per_input += 1
        # `local_connectivity_temperature`
        rand_val: float = 0.01 * torch.randn(1).item()
        self.local_connectivity_probability += rand_val
        if self.local_connectivity_probability < 0:
            self.local_connectivity_probability = 0
        if self.local_connectivity_probability > 1:
            self.local_connectivity_probability = 1

        # ARCHITECTURE PERTURBATION
        # `prune_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if (self.avg_num_prune_mutations % 1) < rand_val:
            num_prune_mutations: An[int, ge(0)] = int(
                self.avg_num_prune_mutations
            )
        else:
            num_prune_mutations: An[int, ge(1)] = (
                int(self.avg_num_prune_mutations) + 1
            )
        for _ in range(num_prune_mutations):
            self.prune_node()
        self.nodes.being_pruned.clear()
        # `grow_node`
        rand_val: An[float, ge(0), le(1)] = float(torch.rand(1))
        if (self.avg_num_grow_mutations % 1) < rand_val:
            num_grow_mutations: An[int, ge(0)] = int(
                self.avg_num_grow_mutations
            )
        else:
            num_grow_mutations: An[int, ge(1)] = (
                int(self.avg_num_grow_mutations) + 1
            )
        starting_node = None
        for _ in range(num_grow_mutations):
            # Chained `grow_node` mutations re-use the previously created
            # hidden node.
            starting_node = self.grow_node(in_node_1=starting_node)

        # NETWORK COMPUTATION COMPONENTS GENERATION
        mutable_nodes: list[Node] = self.nodes.output + self.nodes.hidden
        # A tensor that contains all nodes' in nodes' mutable ids. Used during
        # computation to fetch the correct values from the `outputs` attribute.
        self.in_nodes_indices: Int[Tensor, "NMN MAX_IN"] = -1 * torch.ones(
            (len(mutable_nodes), MAX_INCOMING_CONNECTIONS),
            dtype=torch.int32,
            device=self.device,
        )
        for i, mutable_node in enumerate(mutable_nodes):
            for j, mutable_node_in_node in enumerate(mutable_node.in_nodes):
                self.in_nodes_indices[i][j] = mutable_node_in_node.mutable_uid
        self.weights: Float[Tensor, "NMN MAX_IN"] = torch.tensor(
            self.weights_list, dtype=torch.float32, device=self.device
        )

    def clone(self: "Net") -> "Net":
        # Use state_dict pattern to avoid deepcopy recursion on large graphs.
        # deepcopy on circular node references hits Python's recursion limit
        # after networks grow large (1000+ generations).
        new_net = Net.__new__(Net)
        new_net.load_state_dict(self.get_state_dict())
        return new_net

    def get_state_dict(self: "Net") -> dict:
        # Serialize node structure
        node_states: list[dict] = []
        for node in self.nodes.all:
            node_state = {
                "role": node.role,
                "mutable_uid": node.mutable_uid,
                "immutable_uid": node.immutable_uid,
            }
            if node.role != "input":
                node_state["weights"] = node.weights.copy()
                # Save connections by immutable_uid (stable across mutations)
                node_state["in_node_uids"] = [
                    n.immutable_uid for n in node.in_nodes
                ]
            node_states.append(node_state)

        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "device": self.device,
            "total_num_nodes_grown": self.total_num_nodes_grown,
            "avg_num_grow_mutations": self.avg_num_grow_mutations,
            "avg_num_prune_mutations": self.avg_num_prune_mutations,
            "num_network_passes_per_input": self.num_network_passes_per_input,
            "local_connectivity_probability": self.local_connectivity_probability,
            "node_states": node_states,
            "n_mean_m2_x_z": self.n_mean_m2_x_z.cpu(),  # Move to CPU for pickle
        }

    def load_state_dict(self: "Net", state: dict) -> None:
        # Restore scalar attributes
        self.num_inputs = state["num_inputs"]
        self.num_outputs = state["num_outputs"]
        self.device = state.get("device", "cpu")
        self.total_num_nodes_grown = state["total_num_nodes_grown"]
        self.avg_num_grow_mutations = state["avg_num_grow_mutations"]
        self.avg_num_prune_mutations = state["avg_num_prune_mutations"]
        self.num_network_passes_per_input = state[
            "num_network_passes_per_input"
        ]
        self.local_connectivity_probability = state[
            "local_connectivity_probability"
        ]

        # Restore tensors
        self.n_mean_m2_x_z = state["n_mean_m2_x_z"].to(self.device)

        # Reconstruct node graph (two-pass reconstruction)
        self.nodes = NodeList()
        self.weights_list = []
        uid_to_node: dict[int, Node] = {}  # Map immutable_uid -> Node

        # First pass: create all nodes
        for node_state in state["node_states"]:
            node = Node(
                role=node_state["role"],
                mutable_uid=node_state["mutable_uid"],
                immutable_uid=node_state["immutable_uid"],
            )
            if node.role != "input":
                node.weights = node_state["weights"].copy()

            self.nodes.all.append(node)
            uid_to_node[node.immutable_uid] = node

            if node.role == "input":
                self.nodes.input.append(node)
                self.nodes.receiving.append(node)
            elif node.role == "hidden":
                self.nodes.hidden.append(node)
            elif node.role == "output":
                self.nodes.output.append(node)

        # Second pass: reconnect nodes
        for node_state in state["node_states"]:
            if node_state["role"] != "input":
                node = uid_to_node[node_state["immutable_uid"]]
                for in_uid in node_state["in_node_uids"]:
                    in_node = uid_to_node[in_uid]
                    node.in_nodes.append(in_node)
                    in_node.out_nodes.append(node)
                    # Add once per connection (not once per node)
                    self.nodes.receiving.append(node)
                    self.nodes.emitting.append(in_node)

        # Rebuild weights_list (order must match: output first, then hidden)
        for node in self.nodes.output + self.nodes.hidden:
            self.weights_list.append(node.weights)

        # Rebuild computation components
        mutable_nodes: list[Node] = self.nodes.output + self.nodes.hidden
        self.in_nodes_indices = -1 * torch.ones(
            (len(mutable_nodes), 3), dtype=torch.int32, device=self.device
        )
        for i, node in enumerate(mutable_nodes):
            for j, in_node in enumerate(node.in_nodes):
                self.in_nodes_indices[i][j] = in_node.mutable_uid

        self.weights = torch.tensor(
            self.weights_list, dtype=torch.float32, device=self.device
        )
