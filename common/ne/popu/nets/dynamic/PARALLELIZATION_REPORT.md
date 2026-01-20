# Dynamic Network Mutation Parallelization Report

**Date:** January 2026
**Conclusion:** Parallelization via multiprocessing does NOT help. Sequential is faster.

## Problem Statement

Dynamic networks (`DynamicNets`) have heterogeneous topologies that evolve over time. Unlike static networks (which can use batched tensor operations across all networks), dynamic networks must be mutated individually in a for loop:

```python
for net in self.nets:
    net.mutate()
```

We investigated whether distributing this loop across multiple CPU cores could speed up mutation.

## What We Tried

### Approach 1: multiprocessing.Pool with state dict serialization

**Idea:** Use `Pool.map()` to mutate networks in parallel, serializing/deserializing state dicts each time.

```python
with Pool(processes=num_workers) as pool:
    state_dicts = [net.get_state_dict() for net in self.nets]
    mutated_state_dicts = pool.map(mutate_worker, state_dicts)
    for net, sd in zip(self.nets, mutated_state_dicts):
        net.load_state_dict(sd)
```

**Result:** Much slower than sequential. The serialization overhead on every mutation far exceeded the parallelization benefit.

### Approach 2: Persistent worker pool

**Idea:** Workers own networks persistently, avoiding serialization during mutation. Only serialize during selection/resampling.

```python
# Workers keep networks in memory
# Main process sends "mutate" command
# Workers mutate in-place, return only tensor data needed for forward pass
# Full serialization only during resample()
```

**Result:** Still slower than sequential.

## Detailed Timing Analysis (200 networks, 100 generations)

After warmup (to exclude one-time PyTorch init cost):

| Component | Sequential | Parallel (8 workers) | Speedup |
|-----------|------------|----------------------|---------|
| Mutate core | 32.5ms | 72.3ms | **0.45x** (slower) |
| Prepare | 6.2ms | 5.8ms | 1.07x (same) |
| Resample | 46.2ms | 134.2ms | **0.34x** (slower) |
| **Total** | **84.9ms** | **212.2ms** | **0.40x** (2.5x slower!) |

## Why Parallelization Failed

### 1. Mutation is too fast
- Time per network mutation: ~0.16ms
- IPC overhead (queue send/receive): ~40-70ms per round-trip
- The overhead exceeds the work being parallelized

### 2. Resample requires full serialization
- Selection shuffles networks between workers
- Requires serializing all networks, sending through queues, deserializing
- This cost is unavoidable with the multiprocessing architecture

### 3. PyTorch already parallelizes tensor ops
- `torch.cat`, `torch.bmm`, etc. use multiple CPUs via OpenMP/MKL
- The tensor-heavy parts (`_prepare_for_computation`, `__call__`) already benefit from multi-core
- Only the Python mutation loop is single-threaded, and it's cheap

## One-Time Initialization Cost

We discovered a ~11 second one-time initialization cost on the first call to `_prepare_for_computation()`. This is likely PyTorch JIT compilation or similar. After this, subsequent calls take ~4ms.

This cost was being amortized across generations in our tests, making the per-generation averages look worse than steady-state performance.

## What Actually Helped

### Caching optimization
Instead of accessing network objects during `_prepare_for_computation()`:

```python
# Slow: Python attribute access on each network
for i in range(num_nets):
    num_nodes = len(self.nets[i].nodes.all)
    weights = self.nets[i].weights
    ...
```

We now cache the data after mutation:

```python
# Fast: dict lookups
self._net_data[i] = {
    "num_nodes": len(net.nodes.all),
    "weights": net.weights,
    ...
}
# Then in _prepare_for_computation:
num_nodes = self._net_data[i]["num_nodes"]
```

This made `_prepare_for_computation()` ~20x faster (115ms → 6ms after warmup).

## Bug Fix

During testing, we discovered a bug in `prune_node()`:

```python
# Bug: list.remove() uses equality, could remove wrong weights
self.weights_list.remove(node_being_pruned.weights)

# Fix: use identity comparison
for i, w in enumerate(self.weights_list):
    if w is node_being_pruned.weights:
        del self.weights_list[i]
        break
```

Two nodes can have identical weight values but different weight list objects. The old code would remove the first matching entry, which could be the wrong one.

## Future Considerations

If parallelization is ever needed again, consider:

1. **Shared memory:** Use `multiprocessing.shared_memory` or `torch.multiprocessing` with shared tensors to avoid serialization. Complex with dynamic graph structures.

2. **Workers do everything:** Have workers handle mutation, forward pass, AND evaluation, returning only fitness scores (one float per network). Minimizes serialization but requires major architecture changes.

3. **GPU parallelization:** If mutation becomes expensive (larger networks, more complex operations), GPU-based parallel mutation might help. Would require rewriting mutation logic in tensor operations.

4. **Just accept sequential:** For most practical population sizes (50-500 networks) with small networks, sequential mutation at ~0.16ms/network is fast enough. A 500-network population mutates in ~80ms.

## Final Architecture

The current implementation uses:
- Sequential mutation with data caching
- Batched tensor operations for forward pass (already multi-CPU via PyTorch)
- No explicit multiprocessing

This is simpler and faster than any parallel approach we tried.
