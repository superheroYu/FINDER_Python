"""
FINDER Graph Pool Manager

Implements the same graph pool mechanism as the original FINDER:
- Pre-generate a fixed number of graphs into a graph pool
- Provide random sampling interface
- Support periodic graph pool regeneration
"""

import random
import networkx as nx
from typing import List, Optional, Tuple
import threading
import logging


class GraphPool:
    """
    Graph Pool Manager - Aligned with the original FINDER TrainSet mechanism

    Features:
    1. Pre-generate a specified number of training graphs
    2. Provide random sampling interface
    3. Support thread-safe graph pool updates
    4. Record graph pool usage statistics
    """

    def __init__(
        self,
        min_nodes: int = 30,
        max_nodes: int = 50,
        pool_size: int = 1000,
        graph_type: str = 'barabasi_albert',
        training_type: str = 'uniform',  # Node weight type: 'uniform', 'random', 'degree'
        seed: Optional[int] = None
    ):
        """
        Initialize the graph pool

        Args:
            min_nodes: Minimum number of nodes in a graph
            max_nodes: Maximum number of nodes in a graph
            pool_size: Graph pool size (original FINDER uses 1000)
            graph_type: Graph type
            training_type: Node weight assignment strategy ('uniform'/'random'/'degree')
            seed: Random seed
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.pool_size = pool_size
        self.graph_type = graph_type.replace('-', '_')
        self.training_type = training_type
        self.seed = seed

        # Graph pool storage
        self.graphs: List[nx.Graph] = []
        self.generation_count = 0
        self.sample_count = 0

        # Thread safety
        self._lock = threading.Lock()

        # Logging
        self.logger = logging.getLogger('GraphPool')

        # Initial graph pool generation
        self.generate_new_graphs()

    def generate_graph(self, num_nodes: int) -> nx.Graph:
        """
        Generate a single graph (aligned with the original FINDER gen_graph method)

        Args:
            num_nodes: Number of nodes

        Returns:
            NetworkX graph
        """
        if self.graph_type == 'barabasi_albert':
            # BA model parameters used by the original FINDER
            m = min(4, num_nodes - 1)  # Number of edges per connection
            graph = nx.barabasi_albert_graph(num_nodes, m)
        elif self.graph_type == 'erdos_renyi':
            p = 0.1  # Connection probability
            graph = nx.erdos_renyi_graph(num_nodes, p)
        elif self.graph_type == 'powerlaw':
            # Power-law distribution graph
            graph = nx.powerlaw_cluster_graph(num_nodes, 3, 0.1)
        elif self.graph_type == 'small_world':
            # Small-world network
            k = min(4, num_nodes - 1)
            graph = nx.watts_strogatz_graph(num_nodes, k, 0.1)
        else:
            # Default to BA model
            m = min(4, num_nodes - 1)
            graph = nx.barabasi_albert_graph(num_nodes, m)

        # Ensure the graph is connected
        if not nx.is_connected(graph):
            # Get the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

        # Relabel nodes to consecutive integers
        graph = nx.convert_node_labels_to_integers(graph)

        # Set node weights according to training_type (aligned with original FINDER)
        self._set_node_weights(graph)

        return graph

    def _set_node_weights(self, graph: nx.Graph) -> None:
        """
        Set node weights according to training_type (aligned with original FINDER)

        Args:
            graph: Graph to set weights on
        """
        import numpy as np

        if self.training_type == 'random':
            # Original FINDER: random.uniform(0,1) weights
            weights = {}
            for node in graph.nodes():
                weights[node] = np.random.uniform(0, 1)
        elif self.training_type == 'degree':
            # Original FINDER: degree centrality weights
            degree_centrality = nx.degree_centrality(graph)
            weights = degree_centrality.copy()
        else:  # 'uniform' or other
            # Default: all node weights are 1.0
            weights = {}
            for node in graph.nodes():
                weights[node] = 1.0

        # Set weight attributes
        nx.set_node_attributes(graph, weights, 'weight')

    def generate_new_graphs(self) -> None:
        """
        Generate a new graph pool (aligned with the original FINDER gen_new_graphs method)
        """
        with self._lock:
            self.logger.info(
                f"Generating new training graph pool... (size: {self.pool_size})")

            # Clear existing graph pool
            self.graphs.clear()

            # Generate new graphs
            for i in range(self.pool_size):
                # Randomly select number of nodes
                num_nodes = random.randint(self.min_nodes, self.max_nodes)
                graph = self.generate_graph(num_nodes)
                self.graphs.append(graph)

                # Progress display
                if (i + 1) % 100 == 0:
                    self.logger.debug(
                        f"Generated {i + 1}/{self.pool_size} graphs")

            self.generation_count += 1
            self.logger.info(
                f"Graph pool generation complete! Generation #{self.generation_count}")

    def sample_graph(self) -> nx.Graph:
        """
        Randomly sample a graph from the graph pool (aligned with the original FINDER TrainSet.Sample())

        Returns:
            Deep copied NetworkX graph
        """
        with self._lock:
            if not self.graphs:
                raise ValueError(
                    "Graph pool is empty, please call generate_new_graphs() first")

            # Randomly select a graph
            graph = random.choice(self.graphs)
            self.sample_count += 1

            # Return a deep copy to avoid modifying the original
            return graph.copy()

    def get_pool_stats(self) -> dict:
        """
        Get graph pool statistics

        Returns:
            Statistics dictionary
        """
        with self._lock:
            node_counts = [g.number_of_nodes() for g in self.graphs]
            edge_counts = [g.number_of_edges() for g in self.graphs]

            return {
                'pool_size': len(self.graphs),
                'generation_count': self.generation_count,
                'sample_count': self.sample_count,
                'avg_nodes': sum(node_counts) / len(node_counts) if node_counts else 0,
                'avg_edges': sum(edge_counts) / len(edge_counts) if edge_counts else 0,
                'min_nodes': min(node_counts) if node_counts else 0,
                'max_nodes': max(node_counts) if node_counts else 0,
            }

    def should_regenerate(self, iteration: int, regenerate_interval: int = 5000) -> bool:
        """
        Determine whether the graph pool should be regenerated (original FINDER updates every 5000 iterations)

        Args:
            iteration: Current iteration number
            regenerate_interval: Regeneration interval

        Returns:
            Whether regeneration is needed
        """
        return iteration > 0 and iteration % regenerate_interval == 0


# Global graph pool instances (ensure all environments share the same graph pool)
_global_graph_pools = {}


def get_shared_graph_pool(
    variant: str,
    min_nodes: int = 30,
    max_nodes: int = 50,
    graph_type: str = 'barabasi_albert',
    training_type: str = 'uniform'
) -> GraphPool:
    """
    Get a shared graph pool instance (ensures all environments of the same variant use the same graph pool)

    Args:
        variant: Environment variant name (cn, cn_cost, nd, nd_cost)
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes
        graph_type: Graph type
        training_type: Node weight type

    Returns:
        Shared graph pool instance
    """
    pool_key = f"{variant}_{min_nodes}_{max_nodes}_{graph_type}_{training_type}"

    if pool_key not in _global_graph_pools:
        _global_graph_pools[pool_key] = GraphPool(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            pool_size=1000,  # Consistent with original FINDER
            graph_type=graph_type,
            training_type=training_type
        )

    return _global_graph_pools[pool_key]


def regenerate_all_pools() -> None:
    """
    Regenerate all graph pools (for periodic invocation by the trainer)
    """
    for pool_key, pool in _global_graph_pools.items():
        pool.generate_new_graphs()


def get_all_pool_stats() -> dict:
    """
    Get statistics for all graph pools

    Returns:
        Statistics for all graph pools
    """
    stats = {}
    for pool_key, pool in _global_graph_pools.items():
        stats[pool_key] = pool.get_pool_stats()
    return stats
