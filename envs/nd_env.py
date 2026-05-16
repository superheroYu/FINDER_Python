"""
Network Dismantling Environment (Unweighted)

This module implements the Network Dismantling problem environment for the FINDER framework.
The goal is to select a set of nodes to remove from a graph to minimize the size of the largest connected component in the remaining network.

Based on the original FINDER_ND implementation.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from .base_env import BaseFINDEREnv


class NetworkDismantlingEnv(BaseFINDEREnv):
    """
    Network Dismantling Environment (Unweighted Version)

    In this environment, the agent selects nodes to remove from the graph, aiming to minimize the size of
    the Largest Connected Component (LCC) in the remaining network.

    Unlike the Critical Node problem which uses CND score, Network Dismantling directly optimizes the max connected component size.

    Reward is: -(max connected component size) / (total nodes^2)

    This encourages aggressive fragmentation of the network.
    """

    def __init__(
        self,
        max_nodes: int = 51,  # FINDER standard (NUM_MAX=50 + padding)
        min_nodes: int = 30,
        aux_dim: int = 4,
        seed: Optional[int] = None,
        use_graph_pool: bool = True,
        graph_type: str = 'barabasi_albert',
        training_type: str = 'uniform'  # nd defaults to uniform weights
    ):
        """
        Initialize the Network Dismantling environment.

        Args:
            max_nodes: Maximum number of nodes in the graph
            min_nodes: Minimum number of nodes in the graph
            aux_dim: Auxiliary dimension for additional features
            seed: Random seed for reproducibility
            use_graph_pool: Whether to use the graph pool mechanism
        """
        super().__init__(
            max_nodes, min_nodes, aux_dim, seed,
            variant='nd',
            use_graph_pool=use_graph_pool,
            graph_type=graph_type,
            training_type=training_type
        )

        # Environment-specific state
        self.initial_max_cc_size = 0
        self.current_max_cc_size = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment and compute initial state."""
        observation, info = super().reset(seed=seed, options=options)

        # Compute initial max connected component size
        self.initial_max_cc_size = self._get_max_connected_component_size()
        self.current_max_cc_size = self.initial_max_cc_size

        # Update info with ND-specific metrics
        info.update({
            'initial_max_cc_size': self.initial_max_cc_size,
            'current_max_cc_size': self.current_max_cc_size,
            'cc_reduction': 0,
            'dismantling_ratio': 0.0
        })

        return observation, info

    def _execute_action(self, action: int) -> float:
        """
        Execute action (remove node) and compute reward.

        Uses direct node deletion approach, consistent with base_env.

        Args:
            action: The node index to remove

        Returns:
            Reward for the action
        """
        if action not in self.avail_list:
            return -1.0

        # Directly remove the node
        self.graph.remove_node(action)
        self.removed_nodes.add(action)
        self.avail_list.remove(action)

        # Compute reward
        return self._compute_reward(action)

    def _compute_reward(self, action: int) -> float:
        """
        Compute reward based on max connected component size.

        The reward directly penalizes the size of the largest connected component:

        reward = -(max CC size) / (total nodes^2)

        This encourages the agent to dismantle the network as much as possible,
        reducing the size of the largest remaining component.

        Args:
            action: The action executed (removed node)

        Returns:
            Reward value
        """
        # Update current max connected component size
        self.current_max_cc_size = self._get_max_connected_component_size()

        # Original FINDER ND reward formula normalizes by original node count n
        original_n = self._original_graph.number_of_nodes(
        ) if self._original_graph is not None else self.graph.number_of_nodes()
        if original_n > 0:
            reward = -self.current_max_cc_size / (original_n * original_n)
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

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info including ND-specific metrics."""
        info = super()._get_info()

        # Compute Network Dismantling specific metrics
        cc_reduction = self.initial_max_cc_size - self.current_max_cc_size
        dismantling_ratio = cc_reduction / \
            self.initial_max_cc_size if self.initial_max_cc_size > 0 else 0.0
        efficiency = cc_reduction / \
            len(self.removed_nodes) if self.removed_nodes else 0.0

        # Add Network Dismantling specific info
        info.update({
            'initial_max_cc_size': self.initial_max_cc_size,
            'current_max_cc_size': self.current_max_cc_size,
            'cc_reduction': cc_reduction,
            'dismantling_ratio': dismantling_ratio,
            'dismantling_efficiency': efficiency,
            'nodes_removed_ratio': len(self.removed_nodes) / self._original_graph.number_of_nodes() if self._original_graph else 0.0,
            'problem_type': 'network_dismantling',
            'weighted': False
        })

        return info

    def get_max_connected_nodes_num(self) -> int:
        """
        Get the current size of the largest connected component.

        This method provides compatibility with the original FINDER interface.

        Returns:
            Size of the largest connected component
        """
        return self._get_max_connected_component_size()

    def get_dismantling_ratio(self) -> float:
        """
        Get the current dismantling ratio (reduction in max CC size).

        Returns:
            Ratio of reduction in max connected component size
        """
        if self.initial_max_cc_size == 0:
            return 0.0

        reduction = self.initial_max_cc_size - self.current_max_cc_size
        return reduction / self.initial_max_cc_size

    def get_dismantling_efficiency(self) -> float:
        """
        Get dismantling efficiency (CC size reduction per removed node).

        Returns:
            Average CC size reduction per removed node
        """
        if not self.removed_nodes:
            return 0.0

        reduction = self.initial_max_cc_size - self.current_max_cc_size
        return reduction / len(self.removed_nodes)

    def render(self, mode: str = 'human', **kwargs):
        """Render the environment with ND-specific info."""
        if mode == 'human':
            print(f"=== Network Dismantling Environment ===")
            print(f"Step: {self.current_step}")
            print(f"Remaining nodes: {len(self.avail_list)}")
            print(f"Removed nodes: {len(self.removed_nodes)}")
            print(f"Initial max CC size: {self.initial_max_cc_size}")
            print(f"Current max CC size: {self.current_max_cc_size}")
            reduction = self.initial_max_cc_size - self.current_max_cc_size
            print(
                f"CC reduction: {reduction} ({self.get_dismantling_ratio():.4f})")
            print(
                f"Dismantling efficiency: {self.get_dismantling_efficiency():.4f}")
            print(f"Current step reward: {self.current_step_reward:.4f}")
            print(f"Cumulative reward sum: {self.sum_rewards:.4f}")
            print("=" * 38)
        elif mode == 'matplotlib':
            return self.render_matplotlib(**kwargs)
        else:
            super().render(mode, **kwargs)

    def _get_specific_render_info(self) -> List[str]:
        """Get ND-specific rendering info."""
        info_lines = []
        info_lines.append(f"Initial max CC size: {self.initial_max_cc_size}")
        info_lines.append(f"Current max CC size: {self.current_max_cc_size}")

        reduction = self.initial_max_cc_size - self.current_max_cc_size
        dismantling_ratio = self.get_dismantling_ratio()
        info_lines.append(f"CC reduction: {reduction}")
        info_lines.append(
            f"Dismantling ratio: {dismantling_ratio:.4f} ({dismantling_ratio*100:.2f}%)")
        info_lines.append(
            f"Dismantling efficiency: {self.get_dismantling_efficiency():.4f}")

        # Add connected component info
        if self.graph and self.graph.number_of_nodes() > 0:
            num_components = nx.number_connected_components(self.graph)
            info_lines.append(f"Number of components: {num_components}")

            # Display sizes of connected components
            component_sizes = [len(c)
                               for c in nx.connected_components(self.graph)]
            component_sizes.sort(reverse=True)
            if len(component_sizes) > 1:
                top_sizes = component_sizes[:3] if len(
                    component_sizes) > 3 else component_sizes
                info_lines.append(f"Top component sizes: {top_sizes}")

        info_lines.append(f"Problem type: Network Dismantling (unweighted)")

        return info_lines
