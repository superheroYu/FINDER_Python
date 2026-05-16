English | [中文](README_ZH.md)

# Models and Data Interfaces (Minimal Implementation)

This directory provides a minimal implementation of FINDER policy networks and data interfaces, built around PyTorch + PyTorch Geometric (PyG).

## Components

- `gnn_arch.py`: FinderGNN and loss (including first-order reconstruction term)
- `policy_net.py`: Policy networks (DQN family)
- `data_interfaces.py`: Minimal data interface set
  - `networkx_to_pyg_data(graph, aux_features=None)` → Data
  - `convert_observation_to_pyg_data(obs)` → Data
  - `create_graph_batch_from_observations(observations)` → GraphBatch

GraphBatch fields:
- `node_features`: [N, F]
- `edge_index`: [2, E]
- `batch`: [N]
- `aux_features`: [B, 4]
- `laplacian`: [N, N]
- `edge_weight_sum`: scalar
- `num_graphs`: number of graphs in the batch
- `max_nodes`: maximum node count among graphs in the batch
- `node_ids`: [N]

## Observation Convention

The trainer/environment passes observations via dictionaries:
- Required: `'graph'` (NetworkX graph)
- Optional: `'aux_features'` (np.ndarray(4,); standard values: [covered_ratio, edge_covered_ratio, twohop_density, 1.0])
Other keys are no longer used (e.g. available_actions / action_mask / covered_nodes / current_step).

## Batching

Use `create_graph_batch_from_observations` to directly concatenate observation lists into a GraphBatch.
- If node features are missing, zero tensors of shape [N, 2] are used as a fallback within the batch, ensuring the model forward pass is executable.
- Laplacian is concatenated in block-diagonal form per subgraph.

## Integration with Trainer

The trainer obtains `List[Dict]` observations from `envs.make_batch_env` (internally: gymnasium.vector),
converts them via `data_interfaces` to GraphBatch, and feeds them into the policy network for action selection and training.

## Version

- Minimal implementation; no action masks or available action lists; no environment compatibility check functions.
