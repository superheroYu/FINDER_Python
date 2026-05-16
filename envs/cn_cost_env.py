"""
Critical Node Cost Environment (Weighted)

This module implements the weighted Critical Node problem environment for the FINDER framework.
The goal is to select a set of nodes to remove from a graph to minimize the connectivity of the remaining network,
while accounting for node removal costs/weights.

Based on the original FINDER_CN_cost implementation.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, List
from .base_env import BaseFINDEREnv


class CriticalNodeCostEnv(BaseFINDEREnv):
    """
    Critical Node Cost Environment (Weighted Version)

    In this environment, the agent selects nodes to remove from the graph, aiming to minimize the connectivity of the remaining network,
    while accounting for the cost/weight of removing each node.

    The reward function balances the reduction in max connected component size with the cost of removing nodes:

    reward = -(max connected component size / total nodes) * (node weight / total weight)

    This encourages removing high-impact nodes while considering removal cost.
    """

    def __init__(
        self,
        max_nodes: int = 51,  # FINDER standard (NUM_MAX=50 + padding)
        min_nodes: int = 30,
        aux_dim: int = 4,
        seed: Optional[int] = None,
        weight_range: tuple = (0.0, 1.0),
        use_graph_pool: bool = True,
        graph_type: str = 'barabasi_albert',
        training_type: str = 'random'  # cn_cost defaults to random weights
    ):
        """
        Initialize the Critical Node Cost environment.

        Args:
            max_nodes: Maximum number of nodes in the graph
            min_nodes: Minimum number of nodes in the graph
            aux_dim: Auxiliary dimension for additional features
            seed: Random seed for reproducibility
            weight_range: Random node weight range (min, max)
            use_graph_pool: Whether to use the graph pool mechanism
        """
        super().__init__(
            max_nodes, min_nodes, aux_dim, seed,
            variant='cn_cost',
            use_graph_pool=use_graph_pool,
            graph_type=graph_type,
            training_type=training_type
        )

        # Weight parameters
        self.weight_range = weight_range

        # Node weights and total weight
        self.node_weights = {}
        self.total_node_weight = 0.0

        # Environment-specific state
        self.initial_max_cc_size = 0
        self.current_max_cc_size = 0
        self.total_weight_removed = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment and initialize node weights."""
        observation, info = super().reset(seed=seed, options=options)

        # Initialize node weights
        self._initialize_node_weights(options)

        # Compute initial state
        self.initial_max_cc_size = self._get_max_connected_component_size()
        self.current_max_cc_size = self.initial_max_cc_size
        self.total_weight_removed = 0.0

        # Regenerate observation with weighted features
        observation = self._get_observation()

        # Update info with cost-specific metrics
        info.update({
            'node_weights': self.node_weights.copy(),
            'total_node_weight': self.total_node_weight,
            'initial_max_cc_size': self.initial_max_cc_size,
            'current_max_cc_size': self.current_max_cc_size,
            'total_weight_removed': self.total_weight_removed,
            'average_node_weight': self.total_node_weight / len(self.node_weights) if self.node_weights else 0.0
        })

        return observation, info

    def _initialize_node_weights(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize node weights for the graph.

        Args:
            options: May contain a 'node_weights' key with predefined weights
        """
        # Priority: options > graph pool preset weights > random generation
        if options and 'node_weights' in options:
            # 1. Use weights provided in options (highest priority)
            self.node_weights = options['node_weights'].copy()
        elif self.graph and any('weight' in data for _, data in self.graph.nodes(data=True)):
            # 2. Use preset weights from the graph pool (graph already has weight attributes)
            self.node_weights = {}
            for node, data in self.graph.nodes(data=True):
                self.node_weights[node] = data.get('weight', 1.0)
        else:
            # 3. Generate new random weights (lowest priority, only when graph pool did not preset weights)
            self.node_weights = {}
            for node in self.graph.nodes():
                weight = np.random.uniform(
                    self.weight_range[0], self.weight_range[1])
                self.node_weights[node] = weight

        # Compute total weight
        self.total_node_weight = sum(self.node_weights.values())

        # Weight info will be passed via features attribute, no separate weight attribute needed

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

        # Update total weight removed
        if action in self.node_weights:
            self.total_weight_removed += self.node_weights[action]

        # Directly remove the node
        self.graph.remove_node(action)
        self.removed_nodes.add(action)
        self.avail_list.remove(action)

        # Compute reward
        return self._compute_reward(action)

    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for the weighted critical node problem.

        The reward function considers both the impact on network connectivity and the cost of removing nodes:

        reward = -(max connected component size / total nodes) * (node weight / total weight)

        Args:
            action: The action executed (removed node)

        Returns:
            Reward value
        """
        # Update current max connected component size
        self.current_max_cc_size = self._get_max_connected_component_size()

        # Get node weight
        node_weight = self.node_weights.get(action, 1.0)

        # Compute reward components (per original FINDER: normalized by original node count n)
        original_n = self._original_graph.number_of_nodes(
        ) if self._original_graph is not None else self.graph.number_of_nodes()
        connectivity_ratio = self.current_max_cc_size / \
            original_n if original_n > 0 else 0.0
        weight_ratio = node_weight / \
            self.total_node_weight if self.total_node_weight > 0 else 0.0

        # Original FINDER cost reward formula
        reward = -connectivity_ratio * weight_ratio

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

    def _get_observation(self) -> Dict[str, Any]:
        """
        Get NetworkX format observation including node weight info.

        Returns:
            NetworkX observation dictionary with weight info
        """
        # Get base NetworkX observation
        observation = super()._get_observation()

        # Add weights and features for nodes in the current graph (aligned with original FINDER: node_feat=[weight, 1.0])
        if self.graph and self.node_weights:
            current_graph = observation['graph']
            for node in current_graph.nodes():
                if node in self.node_weights:
                    weight = float(self.node_weights[node])
                    # Node feature order aligned with original implementation: [weight, 1.0]
                    current_graph.nodes[node]['features'] = [weight, 1.0]

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info including cost-specific metrics."""
        info = super()._get_info()

        # Compute additional cost metrics
        cc_reduction = self.initial_max_cc_size - self.current_max_cc_size
        weight_efficiency = cc_reduction / \
            self.total_weight_removed if self.total_weight_removed > 0 else 0.0
        avg_weight_removed = self.total_weight_removed / \
            len(self.removed_nodes) if self.removed_nodes else 0.0

        # Add Critical Node Cost specific info
        info.update({
            'node_weights': self.node_weights.copy(),
            'total_node_weight': self.total_node_weight,
            'total_weight_removed': self.total_weight_removed,
            'weight_removed_ratio': self.total_weight_removed / self.total_node_weight if self.total_node_weight > 0 else 0.0,
            'initial_max_cc_size': self.initial_max_cc_size,
            'current_max_cc_size': self.current_max_cc_size,
            'cc_reduction': cc_reduction,
            'weight_efficiency': weight_efficiency,
            'avg_weight_removed': avg_weight_removed,
            'problem_type': 'critical_node_cost',
            'weighted': True
        })

        return info

    def get_node_weight(self, node: int) -> float:
        """
        Get the weight of a specific node.

        Args:
            node: Node index

        Returns:
            Node weight
        """
        return self.node_weights.get(node, 1.0)

    def get_total_weight(self) -> float:
        """
        Get the total weight of all nodes.

        Returns:
            Total node weight
        """
        return self.total_node_weight

    def get_weight_efficiency(self) -> float:
        """
        Get current weight efficiency (connectivity reduction per unit weight).

        Returns:
            Weight efficiency score
        """
        if self.total_weight_removed <= 0:
            return 0.0

        cc_reduction = self.initial_max_cc_size - self.current_max_cc_size
        return cc_reduction / self.total_weight_removed

    def render(self, mode: str = 'human', **kwargs):
        """Render the environment with cost-specific info."""
        if mode == 'human':
            print(f"=== Critical Node Cost Environment ===")
            print(f"Step: {self.current_step}")
            print(f"Remaining nodes: {len(self.avail_list)}")
            print(f"Removed nodes: {len(self.removed_nodes)}")
            print(f"Total weight: {self.total_node_weight:.4f}")
            print(
                f"Weight removed: {self.total_weight_removed:.4f} ({self.total_weight_removed/self.total_node_weight*100:.2f}%)")
            print(f"Initial max CC size: {self.initial_max_cc_size}")
            print(f"Current max CC size: {self.current_max_cc_size}")
            print(
                f"CC reduction: {self.initial_max_cc_size - self.current_max_cc_size}")
            print(f"Weight efficiency: {self.get_weight_efficiency():.4f}")
            print(f"Current step reward: {self.current_step_reward:.4f}")
            print(f"Cumulative reward sum: {self.sum_rewards:.4f}")
            print("=" * 40)
        elif mode == 'matplotlib':
            return self.render_matplotlib(**kwargs)
        else:
            super().render(mode, **kwargs)

    def _get_specific_render_info(self) -> List[str]:
        """Get CN_cost-specific rendering info."""
        info_lines = []
        info_lines.append(f"Total weight: {self.total_node_weight:.4f}")
        info_lines.append(f"Weight removed: {self.total_weight_removed:.4f}")
        if self.total_node_weight > 0:
            weight_ratio = self.total_weight_removed / self.total_node_weight
            info_lines.append(
                f"Weight removed ratio: {weight_ratio:.4f} ({weight_ratio*100:.2f}%)")

        info_lines.append(f"Initial max CC size: {self.initial_max_cc_size}")
        info_lines.append(f"Current max CC size: {self.current_max_cc_size}")

        cc_reduction = self.initial_max_cc_size - self.current_max_cc_size
        info_lines.append(f"CC reduction: {cc_reduction}")
        info_lines.append(
            f"Weight efficiency: {self.get_weight_efficiency():.4f}")

        # Display weight range info
        info_lines.append(
            f"Weight range: [{self.weight_range[0]:.2f}, {self.weight_range[1]:.2f}]")

        # Add connected component info
        if self.graph and self.graph.number_of_nodes() > 0:
            num_components = nx.number_connected_components(self.graph)
            info_lines.append(f"Number of components: {num_components}")

        info_lines.append(f"Problem type: Critical Node (weighted)")

        return info_lines
