English | [中文](README_ZH.md)

# FINDER Environments

This directory contains Python implementations of FINDER deep reinforcement learning environments for graph-based optimization problems. These environments are converted from the original Cython/C++ implementation to pure Python using NetworkX and gymnasium.

## Overview

The FINDER framework includes four environment variants across two main problems:

### Critical Node (CN) Problem
- **Objective**: Minimize the Connected Component Decomposition (CND) score
- **CND Score**: Sum over all connected components of |C| × (|C| - 1) / 2
- **Environments**:
  - `CriticalNodeEnv`: Unweighted variant
  - `CriticalNodeCostEnv`: Weighted variant (accounts for node removal cost)

### Network Dismantling (ND) Problem
- **Objective**: Minimize the size of the largest connected component
- **Purpose**: Disassemble networks by removing strategic nodes
- **Environments**:
  - `NetworkDismantlingEnv`: Unweighted variant
  - `NetworkDismantlingCostEnv`: Weighted variant (accounts for node removal cost)

## Files

- `base_env.py`: Abstract base class with common functionality
- `cn_env.py`: Critical Node environment (unweighted)
- `cn_cost_env.py`: Critical Node with cost environment (weighted)
- `nd_env.py`: Network Dismantling environment (unweighted)
- `nd_cost_env.py`: Network Dismantling with cost environment (weighted)
- `__init__.py`: Package initialization with factory functions (`make_env`/`make_batch_env`; batching via gymnasium.vector)

## Quick Start

```python
from envs import make_env, make_batch_env
import networkx as nx

# Create a single environment
env = make_env('cn', max_nodes=50, seed=42)

# Create your own graph or let the environment generate one
graph = nx.barabasi_albert_graph(30, 3)

# Reset environment with custom graph
obs, info = env.reset(options={'graph': graph})

# Execute action (remove node)
action = env.random_action()  # or use your RL agent
obs, reward, terminated, truncated, info = env.step(action)

# Render current state
env.render()
```

## Environment Factory

Use factory functions for easy environment creation:

```python
from envs import make_env, make_batch_env

# Single environments
cn_env = make_env('cn', max_nodes=50)
cn_cost_env = make_env('cn_cost', max_nodes=50, weight_range=(0.0, 1.0))
nd_env = make_env('nd', max_nodes=50)
nd_cost_env = make_env('nd_cost', max_nodes=50)

# Batch environment for parallel training (uses gymnasium.vector internally, keeps observations as dicts)
batch_env = make_batch_env('cn', batch_size=64, max_nodes=50)
```

## State Space

- Environments return structured observations in NetworkX format (Dict):
  - `graph`: Current graph (nodes physically removed). Node attribute `features` is `[1.0, 1.0]` in unweighted environments and `[weight, 1.0]` in cost-aware environments.
  - `aux_features`: Length-4 auxiliary features, strictly aligned with original FINDER:
    1. Coverage `|removed| / n_orig`
    2. Edge coverage `( |E_orig| − |E_cur| ) / |E_orig|`
    3. Two-hop density `Σ_v C(deg_v, 2) / n_orig^2`
    4. Constant `1.0`
- Runtime metadata such as `step`, `remaining_nodes`, and `action_list` is returned in the `info` dictionary, not in the observation.

## Action Space

- **Semantics**: Actions are "node IDs" (NetworkX node labels), representing the node to remove
- **Placeholder Space**: For compatibility with gymnasium.vector space checks, the environment internally sets `action_space`/`observation_space` to placeholder `AnySpace(shape=None, dtype=object)`; this does not restrict actual dict observations or action types
- **Validity**: The caller must ensure the selected node exists in the current graph; if an action is invalid, the environment returns a penalty (e.g. `-1.0`) and continues the current episode

## Reward Functions

### Critical Node (CN)
```
reward = - CND_remaining / [ n^2 (n-1) / 2 ]    (n is the original node count)
```

### Critical Node Cost (CN_cost)
```
reward = - ( maxCC / n ) × ( weight(action) / total_weight )
```

### Network Dismantling (ND)
```
reward = - maxCC / n^2
```

### Network Dismantling Cost (ND_cost)
```
reward = - ( maxCC / n ) × ( weight(action) / total_weight )
```

## Custom Graphs and Weights

```python
import random
import networkx as nx

# Create a custom graph
graph = nx.erdos_renyi_graph(40, 0.1)

# For cost environments, specify node weights
node_weights = {node: random.uniform(0.1, 3.0) for node in graph.nodes()}

# Reset with custom graph and weights
options = {
    'graph': graph,
    'node_weights': node_weights  # cost environments only
}
obs, info = env.reset(options=options)
```

## Batching (gymnasium.vector based)

```python
from envs import make_batch_env
import networkx as nx

# Create batch environment (Async/SyncVectorEnv internally)
batch_env = make_batch_env('cn', batch_size=32)

# Create multiple graphs
graphs = [nx.barabasi_albert_graph(35, 3) for _ in range(32)]

# Reset all environments (obs is a list of dicts of length=batch_size; infos is a dict with values aggregated by batch dimension)
obs_batch, infos = batch_env.reset(graphs=graphs)

# Step all environments (done sub-environments are auto-reset)
actions = [0 for _ in range(32)]
obs, rewards, terminated, truncated, infos = batch_env.step(actions)

# Note: infos from vector environments is a "dict" where each key's value is aggregated along the batch dimension (length=batch_size).
# When a sub-environment completes an episode during that step, infos also contains 'final_observation' and 'final_info' keys (also batch-aggregated).
```

## Observation and Placeholder Space Notes (Important)

- To adapt to vectorized environments, `envs/gym_batch.py` internally applies a lightweight wrapper to single environments, only replacing the space with `AnySpace`, without changing the raw dict observations returned by `reset/step`.
- Vector environments return:
  - `obs`: A "list of dict observations" of length batch_size
  - `infos`: A dict (dict-of-arrays), each key's value is a sequence of length=batch_size; when episodes complete, `final_observation`/`final_info` key groups also appear.
- Node feature order for cost variants is `[weight, 1.0]` (in unweighted environments `weight=1.0`).

## Compatibility

- **gymnasium**: Full support for the gymnasium.Env interface
- **NetworkX**: Graph operations via NetworkX
- **NumPy**: NumPy arrays for observations and rewards
- **Vectorization**: Batch environments for parallel processing (gymnasium.vector)

## Integration with RL Frameworks

FINDER observations contain NetworkX graph objects, so use the repository batch adapter instead of constructing `gymnasium.vector.AsyncVectorEnv` directly:

```python
from envs import make_batch_env

vec_env = make_batch_env('cn', batch_size=4, async_env=True)
```

## Performance Notes

- NetworkX operations are reasonably fast for graphs up to ~1000 nodes
- Batch environments process multiple graphs in parallel
- State computation is optimized with caching where possible
- Use smaller graphs (30-50 nodes) for faster training
- Consider optimizing connectivity computation for larger graphs

## API Reference

### Base Environment Parameters
- `max_nodes`: Maximum number of nodes in the graph (default: 51)
- `min_nodes`: Minimum number of nodes in the graph (default: 30)
- `aux_dim`: Auxiliary feature dimension (default: 4)
- `seed`: Random seed (optional)

### Termination Condition Notes
- Original FINDER: no fixed "max removal" limit; termination is based on problem-specific graph coverage/dismantling conditions.
- Current implementation: step truncation is disabled by default (`use_step_truncation = False`). If explicitly enabled, truncation uses `max_steps = max_nodes`.

### Cost Environment Additional Parameters
- `weight_range`: Random node weight range (default: (0.0, 1.0), consistent with original FINDER)

### Main Methods
- `reset()`: Reset environment to initial state
- `step(action)`: Execute action and return new state
- `render()`: Display current environment state
- `random_action()`: Select a random valid action
- `betweenness_action()`: Select action based on betweenness centrality

### Environment-Specific Methods
- `get_robustness(solution)`: Compute robustness metric for a solution
- `get_remaining_cnd_score()`: Get current CND score (CN environments)
- `get_max_connected_nodes_num()`: Get max connected component size (ND environments)
- `get_weight_efficiency()`: Get weight efficiency (cost environments)

## Problem Solving Strategies

Different heuristic strategies can be used for benchmarking:

```python
# Random policy
action = env.random_action()

# Betweenness centrality based policy
action = env.betweenness_action()

# Degree centrality policy (custom implementation)
degrees = {node: env.graph.degree(node) for node in env.avail_list}
action = max(degrees.keys(), key=lambda x: degrees[x])
```
