# Dynamic Networks Module Guide

This module implements neural networks with evolving topology - networks that can grow and prune nodes during evolution, unlike fixed-architecture networks.

## Architecture Overview

```
DynamicNets (base.py)          # Population wrapper - batched computation
├── Net[] (evolution.py)       # Individual network instances
│   ├── NodeList               # Categorized node containers
│   │   ├── input nodes        # Non-parametric, receive observations
│   │   ├── hidden nodes       # Evolved, can grow/prune
│   │   └── output nodes       # Evolved, fixed count
│   └── n_mean_m2_x_z         # Welford stats per node
└── WelfordRunningStandardizer (utils.py)  # Batched normalization
```

## Key Data Structures

### Node (evolution.py:73)

Each node represents a neuron in the graph:

```
role: "input" | "hidden" | "output"
mutable_uid: int      # Position-based ID (changes when nodes added/removed)
immutable_uid: int    # Permanent ID (for serialization/cloning)
in_nodes: list[Node]  # Incoming connections (hidden/output only)
out_nodes: list[Node] # Outgoing connections
weights: list[float]  # MAX_INCOMING_CONNECTIONS=3 weights
```

**Node representation format:**
```
[weights] : in_node_ids → this_node_id → out_node_ids
Example: [-0.86, 0.57, 0.0] : 0-0,1-1 → 6-6 → 4-4
         (node 6 receives from nodes 0,1 and sends to node 4)
```

### NodeList (evolution.py:263)

Groups nodes by role and connection status:
- `all`: All nodes in order
- `input`, `hidden`, `output`: By role
- `receiving`: Nodes with incoming connections (appears once per connection)
- `emitting`: Nodes with outgoing connections (appears once per connection)
- `being_pruned`: Prevents infinite loops during cascading prune

### Net (evolution.py:316)

Individual network with evolving topology:

**Key attributes:**
- `nodes: NodeList` - Graph structure
- `n_mean_m2_x_z: Tensor[NN, 5]` - Welford stats: [n, mean, m2, x, z]
- `weights_list: list[list[float]]` - Mutable weights (for mutation)
- `weights: Tensor` - Weights tensor (for computation)
- `in_nodes_indices: Tensor` - Which nodes feed each mutable node

**Self-adapting parameters:**
- `avg_num_grow_mutations: float` - Expected grow operations per mutate()
- `avg_num_prune_mutations: float` - Expected prune operations per mutate()
- `num_network_passes_per_input: int` - Forward passes per observation (recurrence)
- `local_connectivity_probability: float` - Bias toward local connections

### WelfordRunningStandardizer (utils.py:15)

Online standardization using Welford's algorithm:

```
n_mean_m2_x_z columns:
  [0] n:    observation count
  [1] mean: running mean
  [2] m2:   sum of squared deviations (for variance)
  [3] x:    previous raw value (for detecting new vs old values)
  [4] z:    standardized output
```

**Behavior:**
- `x == 0` or `x == prev_z`: Pass through (no stats update)
- `x != prev_z` and `x != 0`: Update stats, return z-score
- `n < 2`: Return 0 (need 2+ samples for variance)

## How Growth Works

`Net.grow_node()` (evolution.py:400):

1. **During initialization**: Creates input and output nodes (no connections)

2. **Post-initialization** (hidden nodes):
   ```
   Priority 1: Connect non-emitting input nodes (ensure all inputs feed network)
   Priority 2: Connect non-receiving output nodes (ensure all outputs receive)

   Steps:
   1. Sample in_node_1 → new_node (prioritize non-emitting inputs)
   2. Sample in_node_2 → new_node (nearby to in_node_1)
   3. new_node → out_node_1 (prioritize non-receiving outputs)
   ```

3. **Local connectivity bias**: `sample_nearby_node()` uses BFS from the current node, accepting at each distance level with probability `local_connectivity_probability`. Higher values = more modular structures.

## How Pruning Works

`Net.prune_node()` (evolution.py:493):

1. Select a random hidden node to remove
2. Disconnect all its connections
3. **Cascade**: Any hidden node that becomes orphaned (no incoming or outgoing) is also pruned
4. `being_pruned` list prevents infinite recursion

**Important:** Only hidden nodes can be pruned. Input/output nodes are permanent.

## Mutation Flow

`Net.mutate()` (evolution.py:563):

```python
# 1. Parameter perturbation (multiplicative noise)
avg_num_grow_mutations *= (1 + 0.01 * randn)
avg_num_prune_mutations *= (1 + 0.01 * randn)
local_connectivity_probability += 0.01 * randn  # clamped [0,1]
num_network_passes_per_input += ±1 with 1% probability each

# 2. Architecture perturbation
for _ in range(stochastic_count(avg_num_prune_mutations)):
    prune_node()
for _ in range(stochastic_count(avg_num_grow_mutations)):
    grow_node(in_node_1=previous_new_node)  # Chained growth
```

**Chained growth:** Multiple grow operations in one mutation reuse the previously created node as `in_node_1`, creating connected chains rather than isolated nodes.

## Batched Computation (DynamicNets)

`DynamicNets` (base.py) manages multiple `Net` instances with efficient batched computation:

### Index Tensors

Built by `_prepare_for_computation()`:
- `_input_nodes_start_indices[NN]`: Where each net's nodes start in batched tensor
- `_input_nodes_indices[NI×NN]`: Flattened input node positions
- `_output_nodes_indices[NO×NN]`: Flattened output node positions
- `_mutable_nodes_indices[TNMN]`: All non-input nodes
- `_flat_in_nodes_indices[TNMN×3]`: Which nodes feed each mutable node

**Padding trick:** Index 0 is reserved as a dummy node (always outputs 0). Unused connection slots map to index 0.

### Forward Pass (`__call__`)

```python
out = wrs.n_mean_m2_x_z[:, 4].clone()  # Start with previous z-scores
out[input_nodes_indices] = flat_obs    # Insert observations
out = wrs(out)                          # Standardize inputs

for pass_i in range(max_network_passes):
    # Gather inputs for each mutable node
    mapped_out = gather(out, flat_in_nodes_indices).reshape(-1, 3)
    # Weighted sum
    matmuld = (mapped_out * weights).sum(dim=1)
    # Update mutable nodes (masked by pass count)
    out[mutable_nodes_indices] = where(mask[pass_i], matmuld, out[...])
    out = wrs(out)  # Standardize

return out[output_nodes_indices].reshape(num_nets, num_outputs)
```

### Stats Synchronization

**Critical flow:**
1. Forward passes accumulate stats in batched `_wrs.n_mean_m2_x_z`
2. Individual `net.n_mean_m2_x_z` are STALE during forward passes
3. `_update_nets_standardization_values()` syncs stats BACK to individual nets
4. This MUST happen before cloning (mutate or resample)

## Gotchas

### 1. WRS Warmup Period

WRS returns 0 when `n < 2`. For a network with depth D, output nodes need `D+1` forward passes before producing meaningful values:
- Pass 1: Inputs n=1 → return 0
- Pass 2: Inputs n=2 → return z-score, hidden n=1 → return 0
- Pass 3: Hidden n=2 → return z-score, output n=1 → return 0
- Pass 4: Output n=2 → meaningful output

### 2. Cloning Uses state_dict Pattern

`Net.clone()` uses `get_state_dict()`/`load_state_dict()` instead of `deepcopy` because large circular node graphs hit Python's recursion limit (~1000+ generations).

### 3. mutable_uid vs immutable_uid

- `mutable_uid`: Position in node list, changes when nodes are added/removed. Used for tensor indexing.
- `immutable_uid`: Permanent ID assigned at creation. Used for serialization/cloning.

### 4. Weights Order

`weights_list` and `in_nodes_indices` follow the order: **output nodes first, then hidden nodes**. This matches `nodes.output + nodes.hidden`.

### 5. reset() Does NOT Clear WRS Stats

Unlike recurrent networks (which clear hidden states per episode), `DynamicNets.reset()` is a no-op. WRS stats are learned normalizations that should persist.

## File Reference

```
common/ne/popu/nets/dynamic/
├── CLAUDE.md      # This file
├── base.py        # DynamicNets - population wrapper with batched computation
├── evolution.py   # Node, NodeList, Net - single network with topology evolution
└── utils.py       # WelfordRunningStandardizer - online normalization
```

## Testing Changes

```bash
# Quick import test
podman run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    -v "$(pwd):/app" \
    -w /app \
    -e AI_RESEARCH_PATH=/app \
    localhost/maximilienleclei/ai_research:7800xt \
    python -c "from common.ne.popu.nets.dynamic import DynamicNets; print('OK')"

# Run with dynamic nets
podman run ... python -m projects.ne_control_score task=main \
    'popu/nets=dynamic' 'eval.config.env_name=CartPole-v1'
```

## Design Decisions

### MAX_INCOMING_CONNECTIONS = 3

Limits node complexity while allowing non-trivial computation. Empirically chosen balance between expressiveness and search space size.

### Local Connectivity Bias

New connections are biased toward nearby nodes (in graph distance) via `sample_nearby_node()`. This encourages modular network structures rather than random wiring.

### Self-Adapting Mutation Rates

`avg_num_grow_mutations` and `avg_num_prune_mutations` evolve alongside the network. Networks that benefit from more/less growth/pruning will naturally have offspring with adjusted rates.

### Online Standardization

Each node maintains running statistics to standardize its output. This enables stable learning despite topology changes - new nodes adapt their normalization as they accumulate observations.
