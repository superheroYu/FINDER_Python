"""
FINDER Policy Networks - PyTorch Implementation

This module implements policy networks for deep Q-network (DQN) based
reinforcement learning within the FINDER framework using GNN architectures.

Key Features:
- DQN with target network
- Support for Double DQN, Dueling DQN, and other DQN variants
- Prioritized experience replay compatibility
- Graph-based state representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import copy

# Import torch_scatter
from torch_scatter import scatter_mean

from .gnn_arch import FinderGNN, FinderGNNLoss, FinderConfig


class FinderPolicyNetwork(nn.Module):
    """
    FINDER policy network using GNN architecture
    Implements DQN-based policies for graph-structured problems
    """

    def __init__(
        self,
        config: FinderConfig,
        device: torch.device = torch.device('cpu')
    ):
        super(FinderPolicyNetwork, self).__init__()

        self.config = config
        self.device = device

        # Main GNN architecture
        embedding_method = 'graphsage' if config.EMBEDDING_METHOD == 1 else 'structure2vec'
        # GCN uses mean aggregation
        aggregator_map = {0: 'add', 1: 'mean', 2: 'mean'}
        aggregator = aggregator_map[config.AGGREGATOR_ID]

        self.gnn = FinderGNN(
            embedding_size=config.EMBEDDING_SIZE,
            max_bp_iter=config.MAX_BP_ITER,
            reg_hidden=config.REG_HIDDEN,
            aux_dim=config.AUX_DIM,
            embedding_method=embedding_method,
            aggregator=aggregator,
            initialization_stddev=config.INITIALIZATION_STDDEV
        )

        # Loss function
        self.loss_fn = FinderGNNLoss(
            alpha=config.ALPHA,
            use_huber=getattr(config, 'USE_HUBER', False)
        )

        # Target network for stable training
        self.target_gnn = copy.deepcopy(self.gnn)
        self.target_gnn.eval()

        # Freeze target network parameters
        for param in self.target_gnn.parameters():
            param.requires_grad = False

        # Move to device
        self.to(device)

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        use_target: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network

        Args:
            batch_data: Dictionary containing:
                - node_features: [num_nodes, 2]
                - edge_index: [2, num_edges]
                - batch: [num_nodes]
                - action_mask: [batch_size, num_nodes] (optional)
                - aux_features: [batch_size, aux_dim] (optional)
            use_target: Whether to use the target network
            return_embeddings: Whether to return embeddings
        """
        network = self.target_gnn if use_target else self.gnn

        return network(
            node_features=batch_data['node_features'],
            edge_index=batch_data['edge_index'],
            batch=batch_data['batch'],
            act_idxs=batch_data.get('act_idxs'),
            aux_features=batch_data.get('aux_features'),
            return_embeddings=return_embeddings
        )

    def predict_q_values(
        self,
        batch_data: Dict[str, torch.Tensor],
        use_target: bool = False
    ) -> List[np.ndarray]:
        """
        Predict Q-values for available actions in each graph

        Args:
            batch_data: Graph data batch
            available_actions: List of available actions for each graph
            use_target: Whether to use the target network

        Returns:
            List of Q-value arrays for each graph
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # Always select based on Q-values of all nodes
            batch_data_copy = batch_data.copy()
            batch_data_copy.pop('action_mask', None)
            results = self.forward(batch_data_copy, use_target=use_target)
            q_all = results['q_on_all'].cpu().numpy()

            # Slice by graph and return Q-values for all nodes in each graph
            batch_size = batch_data['batch'].max().item() + 1
            predictions = []

            node_idx = 0
            for i in range(batch_size):
                # Compute the number of nodes in this graph
                graph_nodes = (batch_data['batch'] == i).sum().item()
                graph_q_values = q_all[node_idx:node_idx + graph_nodes, 0]
                predictions.append(graph_q_values.copy())
                node_idx += graph_nodes

        self.train(was_training)
        return predictions

    def compute_target_values(
        self,
        next_batch_data: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Vanilla DQN target Q-values: r + gamma * max_a Q_target(s', a).
        Uses the policy's internal target network to ensure the target network
        truly participates in bootstrapping.
        """
        with torch.no_grad():
            next_out = self.forward(
                next_batch_data, use_target=True, return_embeddings=False)
            q_all_next = next_out['q_on_all']
            batch_tensor = next_batch_data['batch']
            batch_size = rewards.size(0)
            next_q_values = torch.zeros(
                batch_size, 1, device=rewards.device, dtype=rewards.dtype)

            for i in range(batch_size):
                mask = (batch_tensor == i)
                if torch.any(mask):
                    next_q_values[i, 0] = q_all_next[mask].max().to(
                        rewards.dtype)

            return rewards + gamma * next_q_values * (1.0 - dones)

    def select_action(
        self,
        batch_data: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
        use_target: bool = False
    ) -> List[int]:
        """
        Select actions using epsilon-greedy policy

        Args:
            batch_data: Graph data batch
            available_actions: Available actions for each graph
            epsilon: Exploration probability
            use_target: Whether to use the target network

        Returns:
            Selected actions for each graph
        """
        batch_tensor = batch_data['batch']
        batch_size = batch_tensor.max().item() + 1
        # Random exploration
        if np.random.random() < epsilon:
            selected = []
            node_idx = 0
            for i in range(batch_size):
                graph_nodes = int((batch_tensor == i).sum().item())
                if 'node_ids' in batch_data:
                    node_ids = batch_data['node_ids']
                    offset = node_idx
                    if graph_nodes > 0:
                        local = np.random.randint(0, graph_nodes)
                        selected.append(int(node_ids[offset + local].item()))
                    else:
                        selected.append(0)
                else:
                    selected.append(np.random.randint(
                        0, max(graph_nodes, 1)) if graph_nodes > 0 else 0)
                node_idx += graph_nodes
            return selected
        # Greedy selection
        q_values = self.predict_q_values(batch_data, use_target)
        selected = []
        node_idx = 0
        for i, q_vals in enumerate(q_values):
            if 'node_ids' in batch_data:
                offset = node_idx
                best_local = int(np.argmax(q_vals)) if len(q_vals) > 0 else 0
                selected.append(
                    int(batch_data['node_ids'][offset + best_local].item()))
            else:
                selected.append(int(np.argmax(q_vals))
                                if len(q_vals) > 0 else 0)
            node_idx += int((batch_tensor == i).sum().item())
        return selected

    def compute_loss(
        self,
        batch_data: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_weights: Optional[torch.Tensor] = None,
        return_td_errors: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss

        Args:
            batch_data: Batch training data
            targets: Target Q-values [batch_size, 1]
            is_weights: Importance sampling weights [batch_size, 1]
            return_td_errors: Whether to return TD errors for prioritized replay

        Returns:
            Dictionary containing loss and optional TD errors
        """
        # Always select corresponding actions based on full-node Q-values, removing dependency on mask
        bd = batch_data.copy()
        bd.pop('action_mask', None)
        results = self.forward(bd, return_embeddings=True)
        node_embeddings = results['node_embeddings']
        q_all = results.get('q_on_all')
        if q_all is None:
            # Degenerate case: if network returned nothing, forward again without mask
            results = self.forward(bd, return_embeddings=True)
            q_all = results['q_on_all']
        # Resolve selected actions for each graph
        batch_tensor = batch_data['batch']
        batch_size = batch_tensor.max().item() + 1
        selected_global = []
        if actions is not None:
            # actions are local graph indices
            node_offset = 0
            for i in range(batch_size):
                graph_nodes = (batch_tensor == i).sum().item()
                a = int(actions[i].item()) if isinstance(
                    actions, torch.Tensor) else int(actions[i])
                a = max(0, min(a, max(graph_nodes - 1, 0)))
                selected_global.append(node_offset + a)
                node_offset += graph_nodes
        else:
            raise ValueError(
                "compute_loss requires actions (per-graph local indices) for selecting regression targets")
        q_pred = q_all[selected_global, :]

        # Compute loss
        loss_results = self.loss_fn(
            q_pred=q_pred,
            targets=targets,
            node_embeddings=node_embeddings,
            laplacian=batch_data['laplacian'],
            edge_weight_sum=batch_data['edge_weight_sum'],
            is_weights=is_weights
        )

        if return_td_errors:
            # Compute TD errors for prioritized experience replay
            td_errors = torch.abs(targets - q_pred).detach().cpu().numpy()
            loss_results['td_errors'] = td_errors

        return loss_results

    def update_target_network(self):
        """Update the target network with current network parameters"""
        self.target_gnn.load_state_dict(self.gnn.state_dict())

    def save_model(self, filepath: str):
        """Save model parameters"""
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'target_gnn_state_dict': self.target_gnn.state_dict(),
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.target_gnn.load_state_dict(checkpoint['target_gnn_state_dict'])


class DoubleDQNPolicyNetwork(FinderPolicyNetwork):
    """
    Double DQN variant of the FINDER policy network
    Reduces overestimation bias in Q-learning
    """

    def compute_target_values(
        self,
        next_batch_data: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute target Q-values using Double DQN

        Args:
            next_batch_data: Next state batch data
            rewards: Immediate rewards [batch_size, 1]
            dones: Done flags [batch_size, 1]
            gamma: Discount factor

        Returns:
            Target Q-values [batch_size, 1]
        """
        with torch.no_grad():
            # Get Q-values from the main network for action selection
            main_q_values = self.predict_q_values(
                next_batch_data, use_target=False)

            # Get Q-values from the target network for value estimation
            target_q_values = self.predict_q_values(
                next_batch_data, use_target=True)

            # Double DQN: use main network to select actions, use target network to evaluate
            batch_size = rewards.size(0)
            next_q_values = torch.zeros(
                batch_size, 1, device=rewards.device, dtype=rewards.dtype)

            for i in range(batch_size):
                done_i = float(dones[i].item())
                if done_i < 0.5 and i < len(main_q_values) and len(main_q_values[i]) > 0:
                    # Use main network to select the best action
                    best_action_idx = np.argmax(main_q_values[i])
                    # Use target network to evaluate
                    next_q_values[i, 0] = float(
                        target_q_values[i][best_action_idx])

            # Compute targets: r + gamma * max_a Q_target(s', a)
            targets = rewards + gamma * next_q_values * (1 - dones)

        return targets


class DuelingDQNPolicyNetwork(DoubleDQNPolicyNetwork):
    """
    Dueling DQN variant with separate value and advantage streams
    Inherits from DoubleDQNPolicyNetwork to gain the overestimation reduction capability of Double DQN
    """

    def __init__(self, config: FinderConfig, device: torch.device = torch.device('cpu')):
        super().__init__(config, device)

        # Graph-level V-value stream: full graph embedding (including auxiliary features) -> multi-layer MLP
        graph_emb_dim = config.EMBEDDING_SIZE + config.AUX_DIM
        # First hidden layer: larger capacity for learning complex mappings
        value_hidden_dim1 = 128
        value_hidden_dim2 = 64   # Second hidden layer: gradual compression
        value_hidden_dim3 = 32   # Third hidden layer: final extraction

        self.value_stream = nn.Sequential(
            # First layer: expand representation space
            nn.Linear(graph_emb_dim, value_hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Second layer: complex feature learning
            nn.Linear(value_hidden_dim1, value_hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Third layer: feature compression
            nn.Linear(value_hidden_dim2, value_hidden_dim3),
            nn.ReLU(),

            # Output layer: value estimation
            nn.Linear(value_hidden_dim3, 1)
        ).to(device)

        # Initialize weights
        for module in self.value_stream:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight, std=config.INITIALIZATION_STDDEV)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.target_value_stream = copy.deepcopy(self.value_stream)
        self.target_value_stream.eval()
        for param in self.target_value_stream.parameters():
            param.requires_grad = False

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        use_target: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using dueling architecture (supports variable-length graphs):
        - Advantage function A(s,a): reuses the original FINDER Q-value computation
        - Value function V(s): computed from graph-level embeddings
        - Final Q-value: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a), where mean is grouped by graph
        """
        network = self.target_gnn if use_target else self.gnn
        base = network(
            node_features=batch_data['node_features'],
            edge_index=batch_data['edge_index'],
            batch=batch_data['batch'],
            act_idxs=None,  # Compute full-node path to obtain per-graph advantage mean
            aux_features=batch_data.get('aux_features'),
            return_embeddings=True
        )

        batch_vec = batch_data['batch']  # [num_nodes]
        num_nodes = batch_vec.size(0)
        batch_size = int(batch_vec.max().item()) + 1 if num_nodes > 0 else 0

        # [B, D + aux_dim] - full graph embedding including auxiliary features
        graph_emb = base['graph_embeddings']

        # Use the original FINDER Q-values as the advantage function A(s,a)
        # [N, 1] - reuse original Q-value computation
        advantages = base['q_on_all']

        # Graph-level V-value estimation: full graph embedding (structure + auxiliary features) -> network
        value_stream = self.target_value_stream if use_target else self.value_stream
        values_graph = value_stream(graph_emb).squeeze(
            1)  # [B] - graph-level V-value estimation
        values_node = values_graph[batch_vec].unsqueeze(
            1)  # [N, 1] - expand back to node level

        # Compute the mean of A within each graph (grouped by batch)
        a_mean_graph = scatter_mean(advantages.squeeze(
            1), batch_vec, dim=0, dim_size=batch_size)  # [B]
        a_mean_node = a_mean_graph[batch_vec].unsqueeze(1)  # [N, 1]

        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q_on_all = values_node + (advantages - a_mean_node)  # [N, 1]

        out: Dict[str, torch.Tensor] = {
            'q_on_all': q_on_all
        }

        # If act_idxs is given, output the corresponding q_pred
        act_idxs = batch_data.get('act_idxs')
        if act_idxs is not None:
            # Map to global indices
            action_indices_list = []
            node_offset = 0
            for i in range(batch_size):
                graph_size = int((batch_vec == i).sum().item())
                local_idx = int(act_idxs[i].item()) if isinstance(
                    act_idxs, torch.Tensor) else int(act_idxs[i])
                if graph_size <= 0:
                    action_indices_list.append(node_offset)
                else:
                    local_idx = max(0, min(local_idx, graph_size - 1))
                    action_indices_list.append(node_offset + local_idx)
                node_offset += graph_size
            action_indices = torch.tensor(
                action_indices_list, device=q_on_all.device, dtype=torch.long)
            out['q_pred'] = q_on_all[action_indices]

        if return_embeddings:
            out['node_embeddings'] = base['node_embeddings']
            out['graph_embeddings'] = base['graph_embeddings']

        return out

    def update_target_network(self):
        """Synchronize the full target network for Dueling DQN."""
        super().update_target_network()
        self.target_value_stream.load_state_dict(
            self.value_stream.state_dict())


# Factory function for creating policy networks
def create_policy_network(
    variant: str,
    config: FinderConfig,
    device: torch.device = torch.device('cpu'),
    dqn_type: str = 'vanilla'
) -> FinderPolicyNetwork:
    """
    Factory function for creating the appropriate policy network

    Args:
        variant: FINDER variant ('CN', 'CN_cost', 'ND', 'ND_cost')
        config: Configuration object
        device: Target device
        dqn_type: DQN type options:
            - 'vanilla': Basic DQN with target network
            - 'double': Double DQN (reduces overestimation bias)
            - 'dueling': Dueling Double DQN (value-advantage decomposition + Double DQN)
            - 'dueling_double': Same as 'dueling' (for backward compatibility)

    Returns:
        Configured policy network
    """
    if dqn_type == 'double':
        return DoubleDQNPolicyNetwork(config, device)
    elif dqn_type in ['dueling', 'dueling_double']:
        # DuelingDQNPolicyNetwork now inherits from DoubleDQNPolicyNetwork,
        # thus automatically combining Dueling and Double DQN techniques
        return DuelingDQNPolicyNetwork(config, device)
    else:
        return FinderPolicyNetwork(config, device)
