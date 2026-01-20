# Neuroevolution Module Guide

This document provides essential context for understanding and modifying the neuroevolution (NE) framework.

## Architecture Overview

```
evolve() loop
├── algo: BaseAlgo          # Selection (e.g., SimpleGA: top 50% duplication)
├── eval: BaseEval          # Fitness evaluation (gym envs, behavior cloning, etc.)
└── popu: BasePopu          # Population wrapper
    └── nets: BaseNets      # Network collection
        └── Net instances   # Individual networks (static or dynamic)
```

**Core loop** (`evolve.py`):
```python
while time < num_minutes:
    popu.nets.mutate()           # Perturb weights/architecture
    fitness = eval(popu, gen)    # Evaluate all networks
    algo(popu, fitness)          # Selection (resample based on fitness)
```

## Key Components

### Networks (popu/nets/)

Two types of networks:

1. **StaticNets** (`static/feedforward.py`, `static/recurrent.py`)
   - Fixed architecture, only weights evolve
   - Vectorized: all networks stored in single tensors with shape `(num_nets, ...)`
   - Self-adaptive mutation: each weight has its own sigma (learning rate)

2. **DynamicNets** (`dynamic/base.py`, `dynamic/evolution.py`)
   - Topology evolves: nodes can grow and be pruned
   - Graph-based: `Node` objects with in/out connections
   - `MAX_INCOMING_CONNECTIONS = 3` per hidden/output node
   - Uses Welford running standardization for stable outputs
   - `n_mean_m2_x_z` tensor: columns are [n, mean, m2, raw_output, standardized_output]

### Evaluators (eval/)

| Evaluator | Fitness Source | Use Case |
|-----------|----------------|----------|
| `GymScoreEval` | Environment rewards | Standard RL tasks |
| `TorchRLScoreEval` | Environment rewards | GPU-heavy envs |
| `ActPredEval` | Action prediction accuracy | Behavior cloning from SB3 |
| `HumanActPredEval` | Action prediction accuracy | Cloning human behavior |
| `AdvGenEval` | Generator + Discriminator fitness | Adversarial imitation |

**Metrics interface**: All evaluators implement `get_metrics() -> dict[str, Tensor]` for logging. Common keys: `env_rewards`, `fitness_G`, `fitness_D`, `target_env_rewards`.

### Populations (popu/)

| Population | Purpose |
|------------|---------|
| `ActorPopu` | Standard action-taking (discrete or continuous) |
| `AdvGenPopu` | Dual-output: actions + discrimination score |

## Important Patterns

### Config Interpolation Pattern

Configs use Hydra interpolation to extract info from evaluators:

```python
@dataclass
class BaseNetsConfig:
    eval: "BaseEval" = "${eval}"  # Hydra resolves this
    num_inputs: int = field(init=False)

    def __post_init__(self):
        self.num_inputs, self.num_outputs = self.eval.retrieve_num_inputs_outputs()
        del self.eval  # Remove to prevent circular serialization
```

**Why `del self.eval`?** The evaluator reference is only needed to extract dimensions. Keeping it would cause circular references during config serialization.

### Shape Conventions

Common shape abbreviations in type hints:
- `NN`: num_nets (population size)
- `NI/NO`: num_inputs/num_outputs
- `NMN`: num_mutable_nodes (hidden + output)
- `TNN/TNMN`: total across all nets (for batched DynamicNets)

### DynamicNets Caching

DynamicNets uses caching for efficient batched computation:

1. `mutate()` calls `_prepare_for_computation()` to build index tensors
2. `_net_data` dict caches per-network info
3. `have_mutated` flag tracks if standardization stats need syncing back

**Important**: Don't call `reset()` expecting it to clear Welford stats - those persist intentionally. `reset()` is for episode-specific state (like RNN hidden states).

### Selection Algorithm

`SimpleGA` implements (μ, λ) selection with μ = λ/2:
1. Sort by fitness descending
2. Take top 50%
3. Duplicate to fill population
4. Call `nets.resample(indices)` to clone networks

## Common Modifications

### Adding a New Evaluator

1. Create class inheriting `BaseEval` in `eval/`
2. Implement:
   - `__call__(population, generation) -> fitness_scores`
   - `retrieve_num_inputs_outputs() -> (int, int)`
   - `retrieve_input_output_specs() -> (obs_spec, action_spec)`
   - `get_metrics() -> dict[str, Tensor]` (for logging)
3. Register in `eval/store.py`

### Adding a New Network Type

1. Create class inheriting `BaseNets` in `popu/nets/`
2. Implement:
   - `mutate()`: modify weights/architecture in-place
   - `resample(indices)`: clone networks by index
   - `__call__(x) -> y`: forward pass
   - `reset()`: clear episode state (optional)
3. Register in `popu/nets/store.py`

## Gotchas

1. **DynamicNets cloning**: Uses `state_dict` pattern instead of `deepcopy` to avoid Python recursion limits on large circular graphs.

2. **Tensor device consistency**: Always check `population.nets.config.device` and move tensors accordingly.

3. **Mutable vs Immutable UIDs**: In DynamicNets, `mutable_uid` changes when nodes are pruned (for tensor indexing), while `immutable_uid` is permanent (for serialization).

4. **Fitness validation**: `evolve()` now validates fitness scores for NaN/Inf. If you hit this, check your evaluator's reward scaling.

5. **Time-based termination**: Evolution runs for `config.num_minutes` wall-clock time, not a fixed number of generations.

## File Reference

```
common/ne/
├── CLAUDE.md      # This file - module documentation for AI agents
├── config.py      # NeuroevolutionTaskConfig, NeuroevolutionSubtaskConfig
├── evolve.py      # Main evolution loop
├── runner.py      # NeuroevolutionTaskRunner (Hydra entry point)
├── store.py       # Config store registration
├── algo/
│   ├── base.py    # BaseAlgo interface
│   ├── simple_ga.py  # 50% truncation selection
│   └── store.py
├── eval/
│   ├── base.py    # BaseEval interface + get_metrics()
│   ├── act_pred.py, adv_gen.py, human_act_pred.py
│   ├── store.py
│   └── score/     # Environment reward evaluators
└── popu/
    ├── base.py    # BasePopu, BasePopuConfig
    ├── actor.py   # ActorPopu for RL
    ├── adv_gen.py # AdvGenPopu for adversarial
    ├── store.py
    └── nets/
        ├── base.py    # BaseNets, BaseNetsConfig
        ├── store.py
        ├── static/    # Fixed-architecture networks
        └── dynamic/   # Topology-evolving networks
            ├── base.py       # DynamicNets (population wrapper)
            ├── evolution.py  # Node, NodeList, Net (single network)
            └── utils.py      # WelfordRunningStandardizer
```

---

## Maintenance Guidelines

### When to Update This Document

Update this CLAUDE.md when you:

1. **Add a new component** (evaluator, algorithm, network type, population)
   - Add to the relevant table/list
   - Document any new patterns or gotchas

2. **Change an interface** (BaseEval, BaseAlgo, BaseNets, BasePopu)
   - Update the "Key Components" section
   - Update "Adding a New X" instructions

3. **Introduce new conventions** (shape abbreviations, config patterns)
   - Add to "Important Patterns" section

4. **Discover a new gotcha**
   - Add to "Gotchas" section with clear explanation

### Code Quality Standards

When modifying this module, maintain:

1. **Type hints**: Every function/method argument including `self`
   ```python
   def mutate(self: "BaseNets") -> None:  # Good
   def mutate(self):  # Bad - missing type hints
   ```

2. **Docstrings**: Google style for all public classes and methods (see common/CLAUDE.md)
   ```python
   def get_metrics(self: "BaseEval") -> dict[str, Tensor]:
       """Return additional metrics from the last evaluation.

       Returns:
           Metric name to tensor value mapping.
       """
   ```

3. **Constants over magic numbers**: Document and name magic values
   ```python
   MAX_INCOMING_CONNECTIONS = 3  # Good - named and documented
   self.weights = [0, 0, 0]      # Bad - magic number
   ```

4. **Metrics interface**: New evaluators must implement `get_metrics()`

### Testing Changes

After modifications, verify:

1. **Import test**: `python -c "from common.ne import evolve"`
2. **Run a simple task**:
   ```bash
   podman run ... python -m projects.ne_control_score task=main
   ```
3. **Check for NaN/Inf**: The validation in `evolve()` will catch these

### Keeping Documentation in Sync

1. **File moves**: Update the "File Reference" tree
2. **New dependencies**: Note in component descriptions
3. **Deprecated patterns**: Move to a "Deprecated" section with migration path
