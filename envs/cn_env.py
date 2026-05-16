"""
Critical Node Environment (Unweighted)

This module implements the Critical Node problem environment for the FINDER framework.
The goal is to select a set of nodes to remove from a graph to minimize the connectivity of the remaining network (measured by CND score).

Based on the original FINDER_CN implementation.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from .base_env import BaseFINDEREnv


class CriticalNodeEnv(BaseFINDEREnv):
    """
    Critical Node Environment (Unweighted Version)

    In this environment, the agent selects nodes to remove from the graph, aiming to minimize the connectivity of the remaining network.
    The reward is based on the Connected Component Decomposition (CND) score.

    CND score = sum over all connected components: |C| * (|C| - 1) / 2
    where |C| is the size of component C.

    The goal is to minimize this score through strategic node removal.
    """

    def __init__(
        self,
        max_nodes: int = 51,  # FINDER standard (NUM_MAX=50 + padding)
        min_nodes: int = 30,
        aux_dim: int = 4,
        seed: Optional[int] = None,
        use_graph_pool: bool = True,
        graph_type: str = 'barabasi_albert',
        training_type: str = 'uniform'  # cn defaults to uniform weights
    ):
        """
        Initialize the Critical Node environment.

        Args:
            max_nodes: Maximum number of nodes in the graph
            min_nodes: Minimum number of nodes in the graph
            aux_dim: Auxiliary dimension for additional features
            seed: Random seed for reproducibility
            use_graph_pool: Whether to use the graph pool mechanism
        """
        super().__init__(
            max_nodes, min_nodes, aux_dim, seed,
            variant='cn',
            use_graph_pool=use_graph_pool,
            graph_type=graph_type,
            training_type=training_type
        )

        # Environment-specific state
        self.initial_cnd_score = 0.0
        self.current_cnd_score = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment and compute initial CND score."""
        observation, info = super().reset(seed=seed, options=options)

        # Compute initial CND score
        self.initial_cnd_score = self._compute_cnd_score()
        self.current_cnd_score = self.initial_cnd_score

        # Update info with CN-specific metrics
        info.update({
            'initial_cnd_score': self.initial_cnd_score,
            'current_cnd_score': self.current_cnd_score,
            'cnd_reduction': 0.0
        })

        return observation, info

    def _execute_action(self, action: int) -> float:
        """
        Execute action (remove node) and compute reward.

        Args:
            action: The node index to remove

        Returns:
            Reward for the action
        """
        # Validate action validity
        if action not in self.graph.nodes():
            return -1.0  # Penalty for invalid action

        # Update FINDER feature 3: neighbor degree sum of removed neighbors
        self._update_neighbor_degree_sum(action)

        # Physically remove the node
        if action in self.graph.nodes():
            self.graph.remove_node(action)

        # Update tracking sets
        self.removed_nodes.add(action)

        # Remove node from available list
        if action in self.avail_list:
            self.avail_list.remove(action)

        # Compute reward based on CND score change
        reward = self._compute_reward(action)

        return reward

    def _compute_reward(self, action: int) -> float:
        """
        Compute reward based on CND score (exactly following the original FINDER_CN implementation).

        Original FINDER_CN reward formula:
        reward = -(remaining CND score) / (n^2 * (n-1) / 2)

        where CND score = sum |C| * (|C| - 1) / 2 (over all connected components C)

        Args:
            action: The action executed (removed node)

        Returns:
            Reward value
        """
        # Compute new CND score after node removal
        new_cnd_score = self._compute_cnd_score()

        # Update current CND score
        self.current_cnd_score = new_cnd_score

        # Original FINDER_CN reward formula (normalized by original graph size)
        original_node_num = self._original_graph.number_of_nodes()
        normalization_factor = original_node_num * \
            original_node_num * (original_node_num - 1) / 2

        if normalization_factor > 0:
            reward = -new_cnd_score / normalization_factor
        else:
            reward = 0.0

        return reward

    def _is_terminal(self) -> bool:
        """
        Consistent with original FINDER: terminates when all edges are covered (logically).
        Under physical deletion implementation, equivalent to the current graph having no edges.
        Returns: True/False
        """
        if not self.graph:
            return True
        return self.graph.number_of_edges() == 0

    def _compute_cnd_score(self) -> float:
        """
        Compute the Connected Component Decomposition (CND) score.

        CND score = sum over all connected components: |C| * (|C| - 1) / 2

        This measures the total number of edges that would exist if each connected component were a complete graph.

        Important implementation update:
        - Directly uses the current graph state (nodes have been physically removed)
        - No need to create subgraphs, improving performance
        - Equivalent to the getRemainingCNDScore() method in the original FINDER C++ implementation

        Returns:
            CND score of the current graph
        """
        if not self.graph or not self.graph.nodes():
            return 0.0

        # Find connected components
        components = list(nx.connected_components(self.graph))

        # Compute CND score
        cnd_score = 0.0
        for component in components:
            size = len(component)
            if size > 1:
                cnd_score += size * (size - 1) / 2

        return cnd_score

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info including CN-specific metrics."""
        info = super()._get_info()

        # Add Critical Node specific info
        cnd_reduction = self.initial_cnd_score - \
            self.current_cnd_score if self.initial_cnd_score > 0 else 0.0
        cnd_reduction_ratio = cnd_reduction / \
            self.initial_cnd_score if self.initial_cnd_score > 0 else 0.0

        info.update({
            'initial_cnd_score': self.initial_cnd_score,
            'current_cnd_score': self.current_cnd_score,
            'cnd_reduction': cnd_reduction,
            'cnd_reduction_ratio': cnd_reduction_ratio,
            'problem_type': 'critical_node',
            'weighted': False
        })

        return info

    def get_remaining_cnd_score(self) -> float:
        """
        Get the current CND score of the remaining graph.

        This method provides compatibility with the original FINDER interface.

        Returns:
            Current CND score
        """
        return self._compute_cnd_score()

    def render(self, mode: str = 'human', **kwargs):
        """Render the environment with CN-specific info."""
        if mode == 'human':
            print(f"=== Critical Node Environment ===")
            print(f"Step: {self.current_step}")
            print(f"Remaining nodes: {len(self.avail_list)}")
            print(f"Removed nodes: {len(self.removed_nodes)}")
            print(f"Initial CND score: {self.initial_cnd_score:.4f}")
            print(f"Current CND score: {self.current_cnd_score:.4f}")
            if self.initial_cnd_score > 0:
                reduction_ratio = (
                    self.initial_cnd_score - self.current_cnd_score) / self.initial_cnd_score
                print(
                    f"CND reduction: {reduction_ratio:.4f} ({reduction_ratio*100:.2f}%)")
            print(f"Current step reward: {self.current_step_reward:.4f}")
            print(f"Cumulative reward sum: {self.sum_rewards:.4f}")
            print(
                f"Max connected component size: {self._get_max_connected_component_size()}")
            print("=" * 35)
        elif mode == 'matplotlib':
            return self.render_matplotlib(**kwargs)
        else:
            super().render(mode, **kwargs)

    def _get_specific_render_info(self) -> List[str]:
        """Get CN-specific rendering info."""
        info_lines = []
        info_lines.append(f"Initial CND score: {self.initial_cnd_score:.4f}")
        info_lines.append(f"Current CND score: {self.current_cnd_score:.4f}")

        if self.initial_cnd_score > 0:
            reduction = self.initial_cnd_score - self.current_cnd_score
            reduction_ratio = reduction / self.initial_cnd_score
            info_lines.append(f"CND reduction: {reduction:.4f}")
            info_lines.append(
                f"Reduction ratio: {reduction_ratio:.4f} ({reduction_ratio*100:.2f}%)")

        # Add connected component info
        if self.graph and self.graph.nodes():
            num_components = nx.number_connected_components(self.graph)
            info_lines.append(f"Number of components: {num_components}")

        info_lines.append(f"Problem type: Critical Node (unweighted)")

        return info_lines
