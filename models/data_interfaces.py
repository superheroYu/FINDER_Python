"""
FINDER Data Interface (Minimal Set):
 - networkx_to_pyg_data: NetworkX -> PyG Data (preserving node_ids and aux_features)
 - convert_observation_to_pyg_data: Single observation Dict -> PyG Data
 - create_graph_batch_from_observations: Observation list -> GraphBatch (for policy networks)
 - GraphBatch: Minimal batch structure required for training
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx


@dataclass
class GraphBatch:
    """
    Minimal graph batch structure for the FINDER model
    """
    # Node features [num_nodes, 2] - usually 2 for FINDER
    node_features: torch.Tensor

    # Edge connectivity [2, num_edges]
    edge_index: torch.Tensor

    # Batch assignment for each node [num_nodes]
    batch: torch.Tensor

    # Auxiliary features for each graph [batch_size, aux_dim]
    aux_features: torch.Tensor

    # Graph Laplacian matrix for reconstruction loss [num_nodes, num_nodes]
    laplacian: torch.Tensor

    # Sum of edge weights (scalar tensor)
    edge_weight_sum: torch.Tensor

    # Metadata
    num_graphs: int = 0
    max_nodes: int = 0
    # Unique mapping: PyG index -> original node ID (aligned with node_features order, concatenated within batch)
    node_ids: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'GraphBatch':
        """Move all tensors to the specified device"""
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            aux_features=self.aux_features.to(device),
            laplacian=self.laplacian.to(device),
            edge_weight_sum=self.edge_weight_sum.to(device),
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
            node_ids=self.node_ids.to(
                device) if self.node_ids is not None else None
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for neural network input"""
        result = {
            'node_features': self.node_features,
            'edge_index': self.edge_index,
            'batch': self.batch,
            'aux_features': self.aux_features,
            'laplacian': self.laplacian,
            'edge_weight_sum': self.edge_weight_sum
        }
        if self.node_ids is not None:
            result['node_ids'] = self.node_ids

        return result


def networkx_to_pyg_data(
    graph: nx.Graph,
    aux_features: Optional[np.ndarray] = None
) -> Data:
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object
    using native PyG conversion utilities.

    Args:
        graph: NetworkX graph
        aux_features: Graph-level auxiliary features
    Returns:
        PyTorch Geometric Data object
    """
    # Ensure every node has FINDER's 2D feature before PyG groups node attrs.
    # Environments normally provide this; external NetworkX graphs may not.
    for node in graph.nodes():
        if 'features' not in graph.nodes[node]:
            graph.nodes[node]['features'] = [1.0, 1.0]

    # Use native PyG conversion tool, specifying node feature attributes.
    data = from_networkx(graph, group_node_attrs=['features'])
    # Unique mapping: record node IDs (PyG index -> original node ID)
    node_id_list = list(graph.nodes())
    data.node_ids = torch.tensor(
        node_id_list, dtype=torch.long) if node_id_list else torch.zeros((0,), dtype=torch.long)

    # Ensure node features exist (FINDER requires 2-dim node features)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Ensure correct node feature mapping (PyG may map 'features' attribute to data.features instead of data.x)
    if num_nodes > 0:
        if hasattr(data, 'features') and not hasattr(data, 'x'):
            # PyG converts 'features' attribute to data.features; we need to map it to data.x
            data.x = data.features
        elif not hasattr(data, 'x'):
            # If no features, use default values
            data.x = torch.ones((num_nodes, 2), dtype=torch.float32)

    # Auxiliary features: use if provided by environment, otherwise not set (handled by upstream/batching stage as fallback)
    if aux_features is not None:
        data.aux_features = torch.tensor(aux_features, dtype=torch.float32)

    # Covered node mask is no longer needed; keep minimal observation and minimal data fields

    # Precompute Laplacian matrix for reconstruction loss
    if num_edges > 0:
        try:
            laplacian_matrix = nx.laplacian_matrix(
                graph, nodelist=node_id_list)
            # Convert to dense tensor for batching
            data.laplacian = torch.tensor(
                laplacian_matrix.toarray(), dtype=torch.float32)
        except Exception:
            data.laplacian = torch.zeros(
                (num_nodes, num_nodes), dtype=torch.float32)
    else:
        data.laplacian = torch.zeros(
            (num_nodes, num_nodes), dtype=torch.float32)

    # Add edge count for reconstruction loss normalization
    data.num_edges_scalar = torch.tensor(float(num_edges), dtype=torch.float32)

    return data


def convert_observation_to_pyg_data(
    observation: Dict[str, Any]
) -> Data:
    """
    Convert a single environment observation dict to PyG Data. The input must
    contain at least 'graph' (NetworkX graph), and may optionally contain
    'aux_features' (np.ndarray(4,)). Returns a torch_geometric.data.Data.
    """
    graph = observation['graph']
    aux_features = observation.get('aux_features')

    # Convert to PyG format
    data = networkx_to_pyg_data(graph, aux_features)

    return data


def create_graph_batch_from_observations(
    observations: List[Dict[str, Any]]
) -> GraphBatch:
    """
    Convert a list of environment observation dicts to a GraphBatch.

    Args:
        observations: List of observation dicts returned by environment reset/step
                      (must contain at least 'graph')

    Returns:
        GraphBatch
    """
    data_list: List[Data] = []
    laplacians: List[torch.Tensor] = []

    # Convert each observation to PyG Data, and extract/separate Laplacians
    for obs in observations:
        data = convert_observation_to_pyg_data(obs)
        laplacians.append(data.laplacian if hasattr(data, 'laplacian') else torch.zeros(
            (data.num_nodes, data.num_nodes), dtype=torch.float32))
        if hasattr(data, 'laplacian'):
            delattr(data, 'laplacian')

        # The weight attribute no longer exists; weight info is passed through features

        data_list.append(data)

    # Create batch
    if data_list:
        batch = Batch.from_data_list(data_list)

        # Node features: fallback to zero tensor [N, 2] if missing
        total_nodes = batch.num_nodes
        node_features = batch.x if (hasattr(batch, 'x') and batch.x is not None) else torch.zeros(
            (total_nodes, 2), dtype=torch.float32)
        edge_index = batch.edge_index
        batch_indices = batch.batch

        aux_features = torch.stack([d.aux_features for d in data_list]) if hasattr(
            data_list[0], 'aux_features') else torch.zeros((len(data_list), 4), dtype=torch.float32)

        # Combine Laplacian block diagonals
        combined_laplacian = torch.zeros(
            (total_nodes, total_nodes), dtype=torch.float32)
        offset = 0
        for i, L in enumerate(laplacians):
            graph_size = (batch_indices == i).sum().item()
            size = min(L.size(0), graph_size)
            combined_laplacian[offset:offset+size,
                               offset:offset+size] = L[:size, :size]
            offset += graph_size

        # The sum of edge weights in the undirected graph adjacency matrix is 2|E|,
        # used to match the original FINDER reconstruction loss normalization.
        total_edges = sum(obs['graph'].number_of_edges()
                          for obs in observations)
        edge_weight_sum = torch.tensor(2.0 * total_edges, dtype=torch.float32)

    else:
        node_features = torch.zeros((0, 2), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch_indices = torch.zeros(0, dtype=torch.long)
        aux_features = torch.zeros((0, 4), dtype=torch.float32)
        combined_laplacian = torch.zeros((0, 0), dtype=torch.float32)
        edge_weight_sum = torch.tensor(0.0, dtype=torch.float32)

    # Combine node_ids (unique mapping channel)
    combined_node_ids = None
    if data_list and hasattr(data_list[0], 'node_ids'):
        combined_node_ids = torch.cat(
            [d.node_ids for d in data_list]) if len(data_list) > 0 else None

    return GraphBatch(
        node_features=node_features,
        edge_index=edge_index,
        batch=batch_indices,
        aux_features=aux_features,
        laplacian=combined_laplacian,
        edge_weight_sum=edge_weight_sum,
        num_graphs=len(observations),
        max_nodes=max(obs['graph'].number_of_nodes()
                      for obs in observations) if observations else 0,
        node_ids=combined_node_ids
    )


def create_graph_batch(
    graphs: List[nx.Graph],
    aux_features_list: Optional[List[np.ndarray]] = None
) -> GraphBatch:
    """
    Create a batched graph representation from a list of NetworkX graphs

    Args:
        graphs: List of NetworkX graphs
        aux_features_list: List of auxiliary features for each graph

    Returns:
        GraphBatch object ready for neural network input
    """
    # Convert each graph to PyG Data - remove Laplacian matrix to avoid dimension mismatch
    data_list = []
    laplacians = []  # Store separately
    for i, graph in enumerate(graphs):
        aux_features = aux_features_list[i] if aux_features_list else None
        data = networkx_to_pyg_data(graph, aux_features)
        # Store Laplacian matrix separately and remove from data
        laplacians.append(data.laplacian)
        delattr(data, 'laplacian')
        data_list.append(data)

    # Create batch
    if data_list:
        batch = Batch.from_data_list(data_list)

        # Extract components
        node_features = batch.x
        edge_index = batch.edge_index
        batch_indices = batch.batch

        # Combine auxiliary features
        aux_features = torch.stack([data.aux_features for data in data_list])

        # Combine Laplacian matrices (block diagonal)
        num_nodes_total = batch.num_nodes
        combined_laplacian = torch.zeros((num_nodes_total, num_nodes_total))

        node_offset = 0
        for i, laplacian in enumerate(laplacians):
            # Get the actual number of nodes in this graph
            num_nodes_in_graph = (batch.batch == i).sum().item()
            # Use the correct size of the Laplacian matrix
            size = min(laplacian.size(0), num_nodes_in_graph)
            combined_laplacian[node_offset:node_offset+size,
                               node_offset:node_offset+size] = laplacian[:size, :size]
            node_offset += num_nodes_in_graph

        # The sum of edge weights in the undirected graph adjacency matrix is 2|E|,
        # used to match the original FINDER reconstruction loss normalization.
        edge_weight_sum = torch.tensor(
            2.0 * sum(len(g.edges()) for g in graphs), dtype=torch.float32)

    else:
        # Empty batch
        node_features = torch.zeros((0, 2), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch_indices = torch.zeros(0, dtype=torch.long)
        aux_features = torch.zeros((0, 4), dtype=torch.float32)
        combined_laplacian = torch.zeros((0, 0), dtype=torch.float32)
        edge_weight_sum = torch.tensor(0.0, dtype=torch.float32)

    # Combine node_ids (unique mapping channel)
    combined_node_ids = None
    if data_list and hasattr(data_list[0], 'node_ids'):
        combined_node_ids = torch.cat(
            [d.node_ids for d in data_list]) if len(data_list) > 0 else None

    return GraphBatch(
        node_features=node_features,
        edge_index=edge_index,
        batch=batch_indices,
        aux_features=aux_features,
        laplacian=combined_laplacian,
        edge_weight_sum=edge_weight_sum,
        num_graphs=len(graphs),
        max_nodes=max(len(g.nodes()) for g in graphs) if graphs else 0,
        node_ids=combined_node_ids
    )


def create_single_graph_batch(
    graph: nx.Graph,
    aux_features: Optional[np.ndarray] = None
) -> GraphBatch:
    """
    Create a batch containing a single graph

    Args:
        graph: NetworkX graph
        aux_features: Auxiliary features

    Returns:
        GraphBatch containing a single graph
    """
    return create_graph_batch(
        graphs=[graph],
        aux_features_list=[aux_features] if aux_features is not None else None
    )
