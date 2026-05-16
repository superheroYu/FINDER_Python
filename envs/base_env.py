"""
FINDER Deep Reinforcement Learning Framework Base Environment Class

This module provides the base abstract class for all FINDER environment variants.
The environment is designed for graph-based optimization problems including critical node and network dismantling problems.

Based on the original Cython/C++ FINDER implementation, providing a pure Python interface compatible with gymnasium standards.
"""

import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import warnings
from .graph_pool import get_shared_graph_pool

# Set matplotlib Chinese font
try:
    # Try common Chinese fonts
    available_fonts = [font.name for font in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS',
                     'PingFang SC', 'Hiragino Sans GB', 'Source Han Sans CN']

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
    else:
        # If no Chinese font found, use system default font
        warnings.warn(
            "No Chinese font found, Chinese characters in visualization may display as boxes")

    plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly
except Exception as e:
    warnings.warn(f"Font setup failed: {e}")


class BaseFINDEREnv(gym.Env, ABC):
    """
    Abstract base class for FINDER environments.

    This class defines the common interface and shared functionality for all FINDER variants:
    - Critical Node (CN): unweighted and weighted versions
    - Network Dismantling (ND): unweighted and weighted versions

    The environment follows the gymnasium interface with graph-specific extensions.
    """

    def __init__(
        self,
        # FINDER uses max_nodes = 51 (NUM_MAX=50 + padding)
        max_nodes: int = 51,
        min_nodes: int = 30,
        aux_dim: int = 4,
        seed: Optional[int] = None,
        # Environment variant, used to get the corresponding graph pool
        variant: str = 'unknown',
        use_graph_pool: bool = True,  # Whether to use the graph pool mechanism
        graph_type: str = 'barabasi_albert',  # Graph type
        training_type: str = 'uniform'  # Node weight type
    ):
        """
        Initialize the base FINDER environment.

        Args:
            max_nodes: Maximum number of nodes in the graph
            min_nodes: Minimum number of nodes in the graph
            aux_dim: Auxiliary dimension for additional features
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Parameter validation
        if max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")
        if min_nodes <= 0:
            raise ValueError(f"min_nodes must be positive, got {min_nodes}")
        if min_nodes > max_nodes:
            # Auto-adjust min_nodes
            min_nodes = max(1, max_nodes // 2)
            print(f"Warning: min_nodes ({min_nodes}) was greater than max_nodes ({max_nodes}). "
                  f"Adjusted min_nodes to {min_nodes}")

        # Store environment parameters
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.aux_dim = aux_dim
        self.variant = variant
        self.use_graph_pool = use_graph_pool
        self.graph_type = graph_type
        self.training_type = training_type

        # Graph pool management (if enabled)
        if self.use_graph_pool:
            self.graph_pool = get_shared_graph_pool(
                variant=variant,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                graph_type=graph_type,
                training_type=training_type
            )
        else:
            self.graph_pool = None

        # Current state variables
        self.graph: Optional[nx.Graph] = None
        self.current_step = 0
        self.max_steps = max_nodes
        # Aligned with original FINDER: step truncation disabled by default
        self.use_step_truncation: bool = False
        self.done = False

        # Action tracking
        self.state_seq: List[List[int]] = []
        self.act_seq: List[int] = []
        self.action_list: List[int] = []
        self.reward_seq: List[float] = []
        self.sum_rewards = 0.0
        self.current_step_reward = 0.0  # Current step reward

        # Graph-specific state
        self.avail_list: List[int] = []
        self.removed_nodes: set = set()  # Track removed nodes
        # Save original node degrees
        self.original_degrees: Dict[int, int] = {}
        # Save a backup of the original graph
        self._original_graph: Optional[nx.Graph] = None

        # Visualization state
        self._fig = None
        self._axes = None
        self._interactive_mode = False

        # Set random seed
        if seed is not None:
            self.seed(seed)

        # --- Custom placeholder Space (shape=None), for compatibility with gym vectorized environments ---
        # Allows arbitrary Python objects to pass through (e.g., dict observations, variable action sets),
        # while satisfying gym's requirement for Space instances.
        class AnySpace(spaces.Space):
            def __init__(self):
                super().__init__(shape=None, dtype=object)

            def sample(self):
                return None

            def contains(self, x) -> bool:
                return True

            def __repr__(self) -> str:
                return "AnySpace(shape=None, dtype=object)"

        # Define action and observation spaces (use placeholder Space to avoid gym.vector errors with None)
        self.action_space = AnySpace()

        # Observation space: use placeholder Space, actual observation is a dict containing NetworkX graph and auxiliary info
        # See get_observation_space() for detailed specification
        self.observation_space = AnySpace()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Random seed
            options: Additional options (may contain a 'graph' key with a NetworkX graph)

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Reset environment state
        self.current_step = 0
        self.done = False
        self.sum_rewards = 0.0
        self.current_step_reward = 0.0

        # Clear sequences
        self.state_seq = []
        self.act_seq = []
        self.action_list = []
        self.reward_seq = []

        # Initialize graph (supports graph pool mechanism)
        if options and 'graph' in options:
            # User-provided custom graph
            self._original_graph = options['graph'].copy()
            self.graph = options['graph'].copy()
        elif self.use_graph_pool and self.graph_pool:
            # Sample from graph pool (aligned with original FINDER)
            self._original_graph = self.graph_pool.sample_graph()
            self.graph = self._original_graph.copy()
        else:
            # Fallback to random generation (if graph pool is disabled)
            self._original_graph = self._generate_random_graph()
            self.graph = self._original_graph.copy()

        # Validate graph
        if self.graph.number_of_nodes() < self.min_nodes or self.graph.number_of_nodes() > self.max_nodes:
            raise ValueError(
                f"Graph must have between {self.min_nodes} and {self.max_nodes} nodes")

        # Initialize original degrees and removed nodes set (FINDER standard)
        self.removed_nodes = set()
        self.original_degrees = {node: self._original_graph.degree(
            node) for node in self._original_graph.nodes()}

        # Compute initial clustering coefficient (cached for performance)
        try:
            self.original_clustering = nx.clustering(self._original_graph)
        except:
            self.original_clustering = {
                node: 0.0 for node in self._original_graph.nodes()}

        # Initialize neighbor degree sum tracking (FINDER feature 3)
        self._neighbor_degree_sum = {
            node: 0.0 for node in self._original_graph.nodes()}

        # Initialize available actions (initially all nodes)
        self.avail_list = list(self.graph.nodes())
        self.action_list = self.avail_list.copy()

        # Compute initial state
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.

        Args:
            action: The action to execute (the node ID to remove/target, referring to the real node label in the NetworkX graph)

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() to restart.")

        if action not in self.avail_list:
            # Invalid action - node already removed or does not exist
            reward = -1.0  # Penalty for invalid action
            terminated = False
            self.current_step_reward = reward  # Record current step reward
        else:
            # Execute action and update node features
            reward = self._execute_action(action)
            self._update_node_features_after_action(action)

            # Record current step reward
            self.current_step_reward = reward

            # Update sequences
            self.act_seq.append(action)
            self.reward_seq.append(reward)
            self.sum_rewards += reward

            # Check termination condition
            terminated = self._is_terminal()

        self.current_step += 1
        # Original FINDER has no fixed step limit; only truncate when explicitly enabled.
        truncated = (self.current_step >=
                     self.max_steps) if self.use_step_truncation else False
        self.done = terminated or truncated

        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = 'human', **kwargs) -> Optional[Any]:
        """
        Render the environment.

        Args:
            mode: Rendering mode ('human', 'rgb_array', 'matplotlib', etc.)
            **kwargs: Additional arguments passed to the visualization function
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Nodes remaining: {len(self.avail_list)}")
            print(f"Removed nodes: {len(self.removed_nodes)}")
            print(f"Current step reward: {self.current_step_reward:.4f}")
            print(f"Cumulative reward: {self.sum_rewards:.4f}")
            print(
                f"Max connected component size: {self._get_max_connected_component_size()}")
            print("---")
        elif mode == 'matplotlib':
            return self.render_matplotlib(**kwargs)
        elif mode == 'rgb_array':
            # Graph visualization can be implemented here
            return None
        else:
            raise NotImplementedError(f"Render mode '{mode}' not implemented")

    @abstractmethod
    def _execute_action(self, action: int) -> float:
        """
        Execute the given action and return the reward.

        Important design update:
        - This implementation now uses physical node deletion instead of logical masking
        - The graph structure (self.graph) is modified after each action
        - removed_nodes and covered_set stay synchronized for compatibility
        - The original graph is preserved in _original_graph for reset use

        Advantages of this design choice:
        1. Clarity: Direct graph structure manipulation, easier to understand and maintain
        2. Performance: Avoids creating subgraphs on every computation
        3. Compatibility: Natural integration with NetworkX observation format
        4. Validation: Passed complete metric consistency testing

        This method is problem-specific and must be implemented by subclasses.

        Args:
            action: The node to remove/target

        Returns:
            Reward for the action
        """
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """
        Check if the environment has reached a terminal state.

        Returns:
            True if terminal, False otherwise
        """
        pass

    @abstractmethod
    def _compute_reward(self, action: int) -> float:
        """
        Compute the reward for a given action.

        Args:
            action: The action executed

        Returns:
            Reward value
        """
        pass

    def _generate_random_graph(self) -> nx.Graph:
        """
        Generate a random graph for training.

        Returns:
            NetworkX graph
        """
        # Generate Barabasi-Albert graph (similar to original FINDER)
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)
        m_edges = min(3, n_nodes - 1)  # Ensure connected graph

        graph = nx.barabasi_albert_graph(n_nodes, m_edges)

        # Ensure the graph is connected
        if not nx.is_connected(graph):
            # Add edges to make it connected
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                u = random.choice(list(components[i]))
                v = random.choice(list(components[i + 1]))
                graph.add_edge(u, v)

        return graph

    def _get_observation(self) -> Dict[str, Any]:
        """
        Get the current observation in NetworkX format (simplified version).

        Returns only the two items needed for training/inference:
        - graph: Current graph state (nodes have been physically removed)
        - aux_features: 4 standard auxiliary features

        Returns:
            Structured observation dictionary
        """
        if self.graph is None:
            # Return empty observation
            empty_graph = nx.Graph()
            return {
                'graph': empty_graph,
                'aux_features': np.zeros(self.aux_dim, dtype=np.float32),
            }

        # Prepare graph data (do not write numpy arrays on node attributes to avoid PyG from_networkx slow path)
        current_graph = self.graph.copy()

        # Add features to nodes (aligned with original FINDER: node_feat=[1.0, 1.0] for unweighted variants)
        for node in current_graph.nodes():
            current_graph.nodes[node]['features'] = [1.0, 1.0]

        # Compute auxiliary features
        aux_features = self._compute_aux_features()

        return {
            'graph': current_graph,
            'aux_features': aux_features
        }

    def _compute_aux_features(self) -> np.ndarray:
        """
        Compute 4-dimensional auxiliary features consistent with original FINDER:
        1) Coverage ratio covered_ratio = |removed_nodes| / n_orig
        2) Edge coverage ratio edge_covered_ratio = |edges incident to removed| / |E_orig|
        3) Two-hop structure quantity twohop_density = twohop_number / (n_orig^2), where twohop_number = sum_v C(deg_v, 2)
        4) Constant bias term 1.0

        Returns:
            4-dimensional auxiliary feature vector (np.float32)
        """
        if self._original_graph is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        n_orig = self._original_graph.number_of_nodes()
        e_orig = self._original_graph.number_of_edges()
        removed_count = len(self.removed_nodes)

        # 1) Coverage ratio
        covered_ratio = (removed_count / n_orig) if n_orig > 0 else 0.0

        # 2) Edge coverage ratio: under physical node deletion, can use (|E_orig| - |E_current|)/|E_orig|
        if self.graph is not None and e_orig > 0:
            e_cur = self.graph.number_of_edges()
            edge_covered_ratio = (e_orig - e_cur) / e_orig
        else:
            edge_covered_ratio = 0.0

        # 3) Two-hop structure quantity: sum_v C(deg_v, 2) / (n_orig^2)
        if self.graph is not None and n_orig > 0:
            twohop_number = 0.0
            for _, d in self.graph.degree():
                if d >= 2:
                    twohop_number += d * (d - 1) / 2.0
            twohop_density = twohop_number / (n_orig * n_orig)
        else:
            twohop_density = 0.0

        # 4) Constant bias
        bias = 1.0

        return np.array([
            float(covered_ratio),
            float(edge_covered_ratio),
            float(twohop_density),
            float(bias)
        ], dtype=np.float32)

    def _update_neighbor_degree_sum(self, removed_node: int):
        """
        Update the neighbor degree sum for removed neighbors (FINDER feature 3).
        When a node is removed, update feature 3 for all its neighbors.

        Args:
            removed_node: The node that was removed
        """
        if self.graph.has_node(removed_node):
            removed_degree = self.original_degrees.get(removed_node, 0)
            for neighbor in self.graph.neighbors(removed_node):
                if neighbor not in self.removed_nodes:  # Only update non-removed neighbors
                    self._neighbor_degree_sum[neighbor] += removed_degree

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.

        Returns:
            Information dictionary
        """
        return {
            'step': self.current_step,
            'remaining_nodes': len(self.avail_list),
            'removed_nodes': len(self.removed_nodes),
            'sum_rewards': self.sum_rewards,
            'max_cc_size': self._get_max_connected_component_size(),
            'action_list': self.action_list.copy(),
            'graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
            'graph_edges': self.graph.number_of_edges() if self.graph else 0
        }

    def _get_max_connected_component_size(self) -> int:
        """
        Get the size of the largest connected component in the current graph.

        Returns:
            Size of the largest connected component
        """
        if not self.graph or not self.graph.nodes():
            return 0

        # Find connected components
        components = list(nx.connected_components(self.graph))
        if not components:
            return 0

        # Return the size of the largest component
        return max(len(comp) for comp in components)

    def get_robustness(self, solution: List[int]) -> float:
        """
        Compute the robustness metric for a given solution.

        Args:
            solution: List of removed node indices

        Returns:
            Robustness score
        """
        if not self.graph:
            return 0.0

        # Create subgraph excluding solution nodes
        remaining_nodes = [n for n in self.graph.nodes() if n not in solution]
        if not remaining_nodes:
            return 1.0

        subgraph = self.graph.subgraph(remaining_nodes)

        # Compute robustness as 1 - (max connected component size / original size)
        if subgraph.number_of_nodes() == 0:
            return 1.0

        components = list(nx.connected_components(subgraph))
        if not components:
            return 1.0

        max_cc_size = max(len(comp) for comp in components)
        original_size = self.graph.number_of_nodes()

        return 1.0 - (max_cc_size / original_size)

    def random_action(self) -> int:
        """
        Select a random valid action.

        Returns:
            Random action (node index)
        """
        if not self.avail_list:
            return 0
        return random.choice(self.avail_list)

    def _update_node_features_after_action(self, action: int):
        """
        Update node features after executing an action (FINDER dynamic feature update).

        Following the original FINDER implementation:
        1. Mark the node as removed (feature 2 = 1.0)
        2. Update feature 3 for all neighbors (neighbor degree sum)

        Args:
            action: The node that was removed
        """
        if action not in self.graph.nodes():
            return

        # 1. Mark the node as removed
        self.removed_nodes.add(action)

        # 2. Get the original degree of the removed node
        removed_node_degree = self.original_degrees.get(action, 0)

        # 3. Update feature 3 for all neighbors (neighbor degree sum)
        for neighbor in self.graph.neighbors(action):
            if neighbor not in self.removed_nodes:  # Only update active neighbors
                if neighbor in self._neighbor_degree_sum:
                    self._neighbor_degree_sum[neighbor] += removed_node_degree
                else:
                    self._neighbor_degree_sum[neighbor] = removed_node_degree

    def render_matplotlib(self,
                          figsize: Tuple[int, int] = (12, 8),
                          node_size_scale: float = 500,
                          pause_time: float = 0.1,
                          save_path: Optional[str] = None,
                          show_legend: bool = True,
                          interactive: bool = True) -> Optional[Any]:
        """
        Visualize the current environment state using matplotlib.

        This is a generic visualization method applicable to all four FINDER environment variants.
        Each variant can override this method to display variant-specific metrics.

        Args:
            figsize: Figure size (width, height)
            node_size_scale: Node size scaling factor
            pause_time: Pause time in interactive mode
            save_path: If provided, save the image to this path
            show_legend: Whether to show the legend
            interactive: Whether to use interactive mode

        Returns:
            Image array if mode is 'rgb_array', otherwise None
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Warning: No graph to display")
            return None

        # Initialize or reuse visualization window
        if interactive:
            if not self._interactive_mode:
                plt.ion()
                self._interactive_mode = True

            # Create or reuse figure
            if self._fig is None or not plt.fignum_exists(self._fig.number):
                self._fig, self._axes = plt.subplots(1, 2, figsize=figsize,
                                                     gridspec_kw={'width_ratios': [3, 1]})
                self._fig.suptitle(
                    'FINDER Environment Visualization', fontsize=16, fontweight='bold')
            else:
                # Clear existing content
                for ax in self._axes:
                    ax.clear()
        else:
            # Non-interactive mode, create a new figure each time
            self._fig, self._axes = plt.subplots(1, 2, figsize=figsize,
                                                 gridspec_kw={'width_ratios': [3, 1]})
            self._fig.suptitle('FINDER Environment Visualization',
                               fontsize=16, fontweight='bold')

        try:
            ax_graph, ax_info = self._axes

            # Draw graph structure
            self._draw_graph(ax_graph, node_size_scale, show_legend)

            # Draw info panel
            self._draw_info_panel(ax_info)

            # Adjust layout
            plt.tight_layout()

            # Save or display
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Image saved to: {save_path}")

            if interactive:
                # Force refresh display
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                if pause_time > 0:
                    plt.pause(pause_time)
            else:
                plt.show()

            return self._fig

        except Exception as e:
            print(f"Error during visualization: {e}")
            return None

    def _draw_graph(self, ax, node_size_scale: float, show_legend: bool):
        """
        Draw the graph structure on the specified axes.

        Args:
            ax: matplotlib axes object
            node_size_scale: Node size scaling factor
            show_legend: Whether to show the legend
        """
        # Use the original graph for layout and visualization, so removed nodes are also shown
        display_graph = self._original_graph if self._original_graph is not None else self.graph

        if display_graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No graph to display', ha='center',
                    va='center', transform=ax.transAxes)
            return

        # Compute graph layout (based on the original graph)
        try:
            pos = nx.spring_layout(display_graph, k=1/np.sqrt(display_graph.number_of_nodes()),
                                   iterations=50, seed=42)
        except:
            pos = nx.random_layout(display_graph, seed=42)

        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []

        for node in display_graph.nodes():
            if node in self.removed_nodes:
                # Removed nodes - red
                node_colors.append('#FF4444')
                # Use original degree to calculate size
                original_degree = self.original_degrees.get(node, 1)
                size = node_size_scale * (0.8 + original_degree / 15)
                node_sizes.append(size)
            elif node in self.avail_list:
                # Available nodes - blue, size based on current degree
                node_colors.append('#4444FF')
                current_degree = self.graph.degree(
                    node) if self.graph.has_node(node) else 0
                size = node_size_scale * (1 + current_degree / 10)
                node_sizes.append(size)
            else:
                # Unavailable but not removed nodes - orange
                node_colors.append('#FF8844')
                current_degree = self.graph.degree(
                    node) if self.graph.has_node(node) else 0
                size = node_size_scale * (0.9 + current_degree / 12)
                node_sizes.append(size)

        # Draw edges - only draw edges that exist in the current graph
        current_edges = []
        edge_colors = []

        for edge in display_graph.edges():
            u, v = edge
            # Only draw edges where both endpoints are still in the current graph
            if (u not in self.removed_nodes and v not in self.removed_nodes and
                    self.graph.has_edge(u, v)):
                current_edges.append(edge)
                edge_colors.append('#333333')
            elif (u in self.removed_nodes or v in self.removed_nodes):
                # Edges involving removed nodes shown in light gray
                current_edges.append(edge)
                edge_colors.append('#CCCCCC')

        # Draw edges
        if current_edges:
            nx.draw_networkx_edges(display_graph, pos, edgelist=current_edges,
                                   ax=ax, edge_color=edge_colors, alpha=0.4, width=1)

        # Draw nodes
        nx.draw_networkx_nodes(display_graph, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # Draw node labels (only for small graphs)
        if display_graph.number_of_nodes() <= 20:
            labels = {}
            for node in display_graph.nodes():
                if node in self.removed_nodes:
                    labels[node] = f"X{node}"  # Removed nodes marked with X
                else:
                    labels[node] = str(node)
            nx.draw_networkx_labels(
                display_graph, pos, labels, ax=ax, font_size=8)

        # Set title and style
        removed_count = len(self.removed_nodes)
        available_count = len(self.avail_list)
        ax.set_title(f'Graph Structure (Step {self.current_step}) - Removed: {removed_count}, Available: {available_count}',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add legend
        if show_legend:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4444FF', alpha=0.8, label='Available'),
                Patch(facecolor='#FF4444', alpha=0.8, label='Removed'),
                Patch(facecolor='#FF8844', alpha=0.8, label='Unavailable'),
                Patch(facecolor='#333333', alpha=0.4, label='Active edge'),
                Patch(facecolor='#CCCCCC', alpha=0.4,
                      label='Disconnected edge')
            ]
            ax.legend(handles=legend_elements,
                      loc='upper right', bbox_to_anchor=(1, 1))

    def _draw_info_panel(self, ax):
        """
        Draw the information panel on the specified axes.

        Args:
            ax: matplotlib axes object
        """
        ax.clear()
        ax.axis('off')

        # Prepare basic info
        info = self._get_info()

        # Basic statistics
        info_text = []
        info_text.append(f"Current step: {self.current_step}")
        info_text.append(f"Total nodes: {info.get('graph_nodes', 0)}")
        info_text.append(f"Total edges: {info.get('graph_edges', 0)}")
        info_text.append(f"Remaining nodes: {len(self.avail_list)}")
        info_text.append(f"Removed nodes: {len(self.removed_nodes)}")
        info_text.append(
            f"Current step reward: {self.current_step_reward:.4f}")
        info_text.append(f"Cumulative reward: {self.sum_rewards:.4f}")
        info_text.append(
            f"Max connected component: {self._get_max_connected_component_size()}")

        # Add environment-specific info (subclasses can override this method)
        specific_info = self._get_specific_render_info()
        if specific_info:
            info_text.append("\n--- Specific Metrics ---")
            info_text.extend(specific_info)

        # Draw text info
        y_pos = 0.95
        for line in info_text:
            ax.text(0.05, y_pos, line, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left')
            y_pos -= 0.08

        # Add progress bar
        self._draw_progress_bar(ax, y_pos - 0.1)

        # Set title
        ax.set_title('Environment Info', fontsize=12,
                     fontweight='bold', pad=20)

    def _get_specific_render_info(self) -> List[str]:
        """
        Get environment-specific rendering information.

        Subclasses should override this method to provide specific metric information.

        Returns:
            List of specific info strings
        """
        return []

    def _draw_progress_bar(self, ax, y_pos: float):
        """
        Draw a progress bar showing step progress.

        Args:
            ax: matplotlib axes object
            y_pos: Y position of the progress bar
        """
        progress = min(self.current_step / self.max_steps, 1.0)

        # Progress bar background
        rect_bg = Rectangle((0.05, y_pos), 0.9, 0.05, transform=ax.transAxes,
                            facecolor='lightgray', edgecolor='black', linewidth=1)
        ax.add_patch(rect_bg)

        # Progress bar fill
        if progress > 0:
            rect_fill = Rectangle((0.05, y_pos), 0.9 * progress, 0.05,
                                  transform=ax.transAxes, facecolor='green', alpha=0.7)
            ax.add_patch(rect_fill)

        # Progress text
        ax.text(0.5, y_pos + 0.025, f'Progress: {progress:.1%}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='center', horizontalalignment='center')

    def close_visualization(self):
        """Close the current visualization window."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
        if self._interactive_mode:
            plt.ioff()
            self._interactive_mode = False
        print("Visualization window closed")

    @staticmethod
    def start_interactive_mode():
        """Start matplotlib interactive mode."""
        plt.ion()
        print("Matplotlib interactive mode started")

    @staticmethod
    def stop_interactive_mode():
        """Stop matplotlib interactive mode."""
        plt.ioff()
        print("Matplotlib interactive mode stopped")

    @staticmethod
    def clear_all_figures():
        """Clear all matplotlib figures."""
        plt.close('all')
        print("All figures cleared")

    def betweenness_action(self) -> int:
        """
        Select an action based on betweenness centrality.

        Returns:
            Node with the highest betweenness centrality
        """
        if not self.avail_list:
            return 0

        # Create subgraph containing available nodes
        subgraph = self.graph.subgraph(self.avail_list)

        if subgraph.number_of_nodes() <= 1:
            return self.avail_list[0] if self.avail_list else 0

        try:
            # Compute betweenness centrality
            centrality = nx.betweenness_centrality(subgraph)
            # Select the node with the highest centrality
            best_node = max(centrality.keys(), key=lambda x: centrality[x])
            return best_node
        except:
            # Fallback to random selection
            return random.choice(self.avail_list)
