"""
FINDER GNN Architecture - PyTorch Implementation

This module converts the original FINDER.pyx BuildNet method from TensorFlow to PyTorch.
It implements graph neural network architectures for critical node detection and
network dismantling problems.

Key Features:
- Structure2Vec and GraphSAGE embedding methods
- Message passing with configurable iterations
- Node-level and graph-level representations
- Support for all four FINDER variants (CN, CN_cost, ND, ND_cost)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


class Structure2VecLayer(MessagePassing):
    """
    Structure2Vec message passing layer implementation
    Based on the original FINDER structure2vec embedding method
    """

    def __init__(self, embed_dim: int, aggr: str = 'add'):
        super(Structure2VecLayer, self).__init__(aggr=aggr)
        self.embed_dim = embed_dim
        self.node_conv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation to neighbor embeddings
        return self.node_conv(x_j)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Structure2Vec: add aggregated messages to original node features
        return F.relu(aggr_out + x)


class GraphSAGELayer(MessagePassing):
    """
    GraphSAGE message passing layer implementation
    Based on the original FINDER graphsage embedding method
    """

    def __init__(self, embed_dim: int, aggr: str = 'mean'):
        super(GraphSAGELayer, self).__init__(aggr=aggr)
        self.embed_dim = embed_dim
        self.neighbor_conv = nn.Linear(embed_dim, embed_dim)
        self.self_conv = nn.Linear(embed_dim, embed_dim)
        self.combine_conv = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation to neighbor embeddings
        return self.neighbor_conv(x_j)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # GraphSAGE: concatenate self and neighbor info, then combine
        self_embed = self.self_conv(x)
        combined = torch.cat([aggr_out, self_embed], dim=1)
        return F.relu(self.combine_conv(combined))


@dataclass
class GraphFeatureEncoderConfig:
    """
    Configuration class for the graph feature encoder
    Independent of FINDER-specific configuration, focused on GNN feature encoding parameters
    """

    # Basic architecture parameters
    embedding_size: int = 64
    max_bp_iter: int = 3
    embedding_method: str = 'graphsage'  # 'structure2vec' or 'graphsage'
    aggregator: str = 'sum'  # 'sum', 'mean', or 'gcn'
    initialization_stddev: float = 0.01

    # Input feature dimension
    input_dim: int = 2  # Original FINDER node feature dimension

    # Whether to apply L2 normalization after each layer
    layer_norm: bool = True

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.embedding_method not in ['structure2vec', 'graphsage']:
            raise ValueError(
                f"Unsupported embedding method: {self.embedding_method}")

        if self.aggregator not in ['sum', 'mean', 'gcn', 'add']:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

        # Normalize aggregator name: map 'add' to 'sum'
        if self.aggregator == 'add':
            self.aggregator = 'sum'

        if self.embedding_size <= 0 or self.max_bp_iter <= 0:
            raise ValueError("Embedding size and max_bp_iter must be positive")


class GraphFeatureEncoder(nn.Module):
    """
    Standalone graph feature encoder

    Responsible for generating node/graph embeddings from graph structure and node features,
    without any Q-value computation logic. Supports both Structure2Vec and GraphSAGE
    feature encoding methods.

    Core Design Principles:
    - Single Responsibility: only responsible for graph encoding
    - Reusable: independent of FINDER-specific logic
    - Extensible: easy to add new encoding methods

    Usage Example:
    ```python
    config = GraphFeatureEncoderConfig(
        embedding_size=64,
        max_bp_iter=3,
        embedding_method='graphsage'
    )
    encoder = GraphFeatureEncoder(config)

    # Encode graph data
    result = encoder(node_features, edge_index, batch)
    node_embeds = result['node_embeddings']
    graph_embeds = result['graph_embeddings']
    ```
    """

    def __init__(self, config: GraphFeatureEncoderConfig):
        super(GraphFeatureEncoder, self).__init__()

        self.config = config
        self.embedding_size = config.embedding_size
        self.max_bp_iter = config.max_bp_iter
        self.embedding_method = config.embedding_method
        self.aggregator = config.aggregator
        self.layer_norm = config.layer_norm

        # Initial node embedding layer (equivalent to w_n2l in the original)
        self.initial_embed = nn.Linear(config.input_dim, config.embedding_size)

        # Message passing layers
        self.mp_layers = nn.ModuleList()

        # PyG MessagePassing expects 'add' instead of 'sum'
        pyg_aggregator = 'add' if config.aggregator == 'sum' else config.aggregator

        for _ in range(config.max_bp_iter):
            if config.embedding_method == 'structure2vec':
                self.mp_layers.append(Structure2VecLayer(
                    config.embedding_size, aggr=pyg_aggregator))
            elif config.embedding_method == 'graphsage':
                self.mp_layers.append(GraphSAGELayer(
                    config.embedding_size, aggr=pyg_aggregator))
            else:
                raise ValueError(
                    f"Unknown embedding method: {config.embedding_method}")

        # Initialize weights
        self._initialize_weights(config.initialization_stddev)

    def _initialize_weights(self, stddev: float):
        """Initialize network weights using truncated normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=stddev)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the graph feature encoder

        Args:
            node_features: Node feature tensor [num_nodes, input_dim]
            edge_index: COO edge index [2, num_edges]
            batch: Node-to-graph batch assignment vector [num_nodes], values in [0, batch_size-1]

        Returns:
            Dict containing:
            - node_embeddings: Node embeddings [num_nodes, embed_dim]
            - graph_embeddings: Graph-level embeddings [batch_size, embed_dim]
        """
        # Initial node embeddings
        x = F.relu(self.initial_embed(node_features))
        if self.layer_norm:
            # L2 normalization, as in the original
            x = F.normalize(x, p=2, dim=1)

        # Message passing iterations
        for layer in self.mp_layers:
            x = layer(x, edge_index)
            if self.layer_norm:
                # L2 normalization after each layer
                x = F.normalize(x, p=2, dim=1)

        # Store final node embeddings
        node_embeddings = x

        # Graph-level representation (pooling)
        graph_embeddings = global_add_pool(x, batch)  # [batch_size, embed_dim]

        return {
            'node_embeddings': node_embeddings,
            'graph_embeddings': graph_embeddings
        }

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension"""
        return self.embedding_size

    def get_config(self) -> GraphFeatureEncoderConfig:
        """Return the feature encoder configuration"""
        return self.config


class FinderGNN(nn.Module):
    """
    Main FINDER graph neural network architecture
    Converts the original FINDER.pyx BuildNet method to PyTorch
    """

    def __init__(
        self,
        embedding_size: int = 64,
        max_bp_iter: int = 3,
        reg_hidden: int = 32,
        aux_dim: int = 4,
        embedding_method: str = 'graphsage',  # 'structure2vec' or 'graphsage'
        aggregator: str = 'sum',  # 'sum', 'mean', or 'gcn'
        initialization_stddev: float = 0.01
    ):
        super(FinderGNN, self).__init__()

        # Save parameters for backward compatibility
        self.embedding_size = embedding_size
        self.max_bp_iter = max_bp_iter
        self.reg_hidden = reg_hidden
        self.aux_dim = aux_dim
        self.embedding_method = embedding_method
        self.aggregator = aggregator

        # Create graph feature encoder configuration
        encoder_config = GraphFeatureEncoderConfig(
            embedding_size=embedding_size,
            max_bp_iter=max_bp_iter,
            embedding_method=embedding_method,
            aggregator=aggregator,
            initialization_stddev=initialization_stddev,
            input_dim=2,  # FINDER uses 2-dim node features
            layer_norm=True
        )

        # Graph feature encoder (replaces the original embedding and message passing layers)
        self.graph_encoder = GraphFeatureEncoder(encoder_config)

        # Cross product layer for state-action embeddings
        self.cross_product = nn.Linear(embedding_size, 1)

        # Regression layers for Q-value prediction
        # Note: The original FINDER always uses auxiliary features; we remain consistent here
        if reg_hidden > 0:
            self.h1_weight = nn.Linear(embedding_size, reg_hidden)
            self.h2_weight = nn.Linear(reg_hidden + aux_dim, 1)
        else:
            self.output_layer = nn.Linear(embedding_size + aux_dim, 1)

        # Save aux_dim for runtime checking
        self._expected_aux_dim = aux_dim

        # Initialize weights for Q-value computation layers
        self._initialize_q_layers(initialization_stddev)

    def _initialize_q_layers(self, stddev: float):
        """Initialize weights for Q-value computation layers (graph encoder weights are already initialized internally)"""
        modules_to_init = [self.cross_product]

        if self.reg_hidden > 0:
            modules_to_init.extend([self.h1_weight, self.h2_weight])
        else:
            modules_to_init.append(self.output_layer)

        for module in modules_to_init:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=stddev)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        act_idxs: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the FINDER GNN

        Args:
            - node_features: Node feature tensor [num_nodes, 2] (originally 1)
            - edge_index: COO edge index [2, num_edges]
            - batch: Node-to-graph batch assignment vector [num_nodes], values in [0, batch_size-1]
            - act_idxs: Local node indices of selected actions in each graph [batch_size];
                        when None, score all nodes
            - aux_features: Graph-level auxiliary features [batch_size, aux_dim] (optional)
            - return_embeddings: Whether to return node/graph embeddings in the result

        Returns:
            When act_idxs is not None, returns:
            - q_pred: Q-values for selected actions [batch_size, 1]
            - embed_s_a: State-action joint embeddings [batch_size, embed_dim]

            When act_idxs is None, returns:
            - q_on_all: Q-values for all nodes [num_nodes, 1]
            - embed_s_a_all: State-action joint embeddings for all nodes [num_nodes, embed_dim]

            If return_embeddings is True, additionally returns:
            - node_embeddings: Node embeddings [num_nodes, embed_dim]
            - graph_embeddings: Graph-level embeddings [batch_size, embed_dim]
        """
        batch_size = batch.max().item() + 1

        # Process auxiliary features: use zero vectors if not provided (consistent with original FINDER)
        if aux_features is None:
            aux_features = torch.zeros(batch_size, self._expected_aux_dim,
                                       device=node_features.device, dtype=node_features.dtype)

        # Use graph feature encoder to get node and graph embeddings
        encoder_output = self.graph_encoder(node_features, edge_index, batch)
        node_embeddings = encoder_output['node_embeddings']
        graph_embeddings = encoder_output['graph_embeddings']

        results = {}

        if act_idxs is not None:
            # Build global node indices based on per-graph local indices
            action_indices_list = []
            node_offset = 0
            for i in range(batch_size):
                graph_size = (batch == i).sum().item()
                local_idx = int(act_idxs[i].item()) if isinstance(
                    act_idxs, torch.Tensor) else int(act_idxs[i])
                if graph_size <= 0:
                    action_indices_list.append(node_offset)
                else:
                    local_idx = max(0, min(local_idx, graph_size - 1))
                    action_indices_list.append(node_offset + local_idx)
                node_offset += graph_size
            action_indices = torch.tensor(
                action_indices_list, device=node_embeddings.device, dtype=torch.long)
            # [batch_size, embed_dim]
            action_embeddings = node_embeddings[action_indices]

            # Cross product between action and state embeddings
            # Equivalent to the matrix multiplication in original FINDER
            temp = torch.bmm(
                action_embeddings.unsqueeze(2),  # [batch_size, embed_dim, 1]
                graph_embeddings.unsqueeze(1)   # [batch_size, 1, embed_dim]
            )  # [batch_size, embed_dim, embed_dim]

            # Apply cross product transformation
            cross_weights = self.cross_product.weight.t().expand(
                batch_size, -1, -1)  # Transpose weight and expand
            embed_s_a = torch.bmm(temp, cross_weights).squeeze(
                2)  # [batch_size, embed_dim]

            # Regression layers
            if self.reg_hidden > 0:
                hidden = F.relu(self.h1_weight(embed_s_a))
                if aux_features is not None:
                    hidden = torch.cat([hidden, aux_features], dim=1)
                q_pred = self.h2_weight(hidden)
            else:
                if aux_features is not None:
                    embed_s_a = torch.cat([embed_s_a, aux_features], dim=1)
                q_pred = self.output_layer(embed_s_a)

            # Expose intermediate state-action embeddings for external policies (e.g., Dueling) to use
            results['q_pred'] = q_pred
            results['embed_s_a'] = embed_s_a

        else:
            # Compute Q-values for all nodes
            # Repeat graph embeddings for each node
            num_nodes = node_embeddings.size(0)
            # [num_nodes, embed_dim]
            graph_rep_expanded = graph_embeddings[batch]

            # Cross product for all nodes
            temp = torch.bmm(
                node_embeddings.unsqueeze(2),  # [num_nodes, embed_dim, 1]
                graph_rep_expanded.unsqueeze(1)  # [num_nodes, 1, embed_dim]
            )  # [num_nodes, embed_dim, embed_dim]

            # Apply cross product transformation
            cross_weights = self.cross_product.weight.t().expand(
                num_nodes, -1, -1)  # Transpose weight and expand
            embed_s_a_all = torch.bmm(temp, cross_weights).squeeze(
                2)  # [num_nodes, embed_dim]

            # Regression layers
            if self.reg_hidden > 0:
                hidden = F.relu(self.h1_weight(embed_s_a_all))
                # Expand aux features for all nodes
                if aux_features is not None:
                    aux_expanded = aux_features[batch]  # [num_nodes, aux_dim]
                    hidden = torch.cat([hidden, aux_expanded], dim=1)
                q_on_all = self.h2_weight(hidden)
            else:
                if aux_features is not None:
                    aux_expanded = aux_features[batch]
                    embed_s_a_all = torch.cat(
                        [embed_s_a_all, aux_expanded], dim=1)
                q_on_all = self.output_layer(embed_s_a_all)

            results['q_on_all'] = q_on_all
            results['embed_s_a_all'] = embed_s_a_all

        if return_embeddings:
            results['node_embeddings'] = node_embeddings
            # Return the complete graph state representation (including auxiliary features)
            # This is consistent with the state representation used in Q-value computation
            complete_graph_embeddings = torch.cat(
                [graph_embeddings, aux_features], dim=1)
            results['graph_embeddings'] = complete_graph_embeddings
            # Also provide pure graph structure embeddings for scenarios that need them
            results['graph_structure_embeddings'] = graph_embeddings

        return results

    def get_graph_encoder(self) -> GraphFeatureEncoder:
        """Get the underlying graph feature encoder for debugging and analysis"""
        return self.graph_encoder

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Perform graph encoding only, without computing Q-values (for standalone use of the encoder)"""
        return self.graph_encoder(node_features, edge_index, batch)


class FinderGNNLoss(nn.Module):
    """
    Loss function for the FINDER GNN, including reconstruction loss
    Implements RL loss and graph reconstruction loss, as in the original version
    """

    def __init__(self, alpha: float = 0.001, use_huber: bool = False):
        super(FinderGNNLoss, self).__init__()
        self.alpha = alpha  # Weight for reconstruction loss
        self.use_huber = use_huber

        if use_huber:
            self.rl_loss = nn.SmoothL1Loss()
        else:
            self.rl_loss = nn.MSELoss()

    def forward(
        self,
        q_pred: torch.Tensor,
        targets: torch.Tensor,
        node_embeddings: torch.Tensor,
        laplacian: torch.Tensor,
        edge_weight_sum: torch.Tensor,
        is_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: RL loss + reconstruction loss

        Args:
            q_pred: Predicted Q-values [batch_size, 1]
            targets: Target Q-values [batch_size, 1]
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            laplacian: Graph Laplacian matrix [num_nodes, num_nodes]
            edge_weight_sum: Sum of edge weights (scalar)
            is_weights: Importance sampling weights [batch_size, 1] (optional)
        """
        # RL loss (Q-learning loss)
        if is_weights is not None:
            # Prioritized experience replay: normalize by sum of weights to prevent
            # batch-to-batch loss scale drift with average weight
            td_errors = targets - q_pred
            denom = torch.clamp(is_weights.sum(), min=1e-8)
            rl_loss = (is_weights * td_errors.pow(2)).sum() / denom
        else:
            rl_loss = self.rl_loss(q_pred, targets)

        # First-order graph reconstruction loss
        # Equivalent to: 2 * trace(X^T * L * X) / edge_weight_sum
        safe_edge_weight_sum = torch.clamp(edge_weight_sum, min=1.0)
        reconstruction_loss = 2 * torch.trace(
            torch.mm(
                torch.mm(node_embeddings.t(), laplacian),
                node_embeddings
            )
        ) / safe_edge_weight_sum

        # Combined loss
        total_loss = rl_loss + self.alpha * reconstruction_loss

        return {
            'total_loss': total_loss,
            'rl_loss': rl_loss,
            'reconstruction_loss': reconstruction_loss
        }


# Configuration classes for different FINDER variants
class FinderConfig:
    """Base configuration for FINDER variants"""

    # Hyperparameters from the original FINDER
    GAMMA = 1.0  # Discount factor
    EMBEDDING_SIZE = 64
    LEARNING_RATE = 0.0001
    REG_HIDDEN = 32
    AUX_DIM = 4
    MAX_BP_ITER = 3
    AGGREGATOR_ID = 0  # 0: sum, 1: mean, 2: GCN
    EMBEDDING_METHOD = 1  # 0: structure2vec, 1: graphsage
    INITIALIZATION_STDDEV = 0.01
    ALPHA = 0.001  # Reconstruction loss weight
    USE_HUBER = False  # Whether to use Huber loss

    @classmethod
    def create_graph_encoder_config(cls) -> GraphFeatureEncoderConfig:
        """
        Create a graph feature encoder configuration from the FINDER configuration,
        facilitating configuration consistency.
        """
        # Convert aggregator ID to string
        aggregator_map = {0: 'sum', 1: 'mean', 2: 'gcn'}
        aggregator = aggregator_map.get(cls.AGGREGATOR_ID, 'sum')

        # Convert embedding method ID to string
        method_map = {0: 'structure2vec', 1: 'graphsage'}
        embedding_method = method_map.get(cls.EMBEDDING_METHOD, 'graphsage')

        return GraphFeatureEncoderConfig(
            embedding_size=cls.EMBEDDING_SIZE,
            max_bp_iter=cls.MAX_BP_ITER,
            embedding_method=embedding_method,
            aggregator=aggregator,
            initialization_stddev=cls.INITIALIZATION_STDDEV,
            input_dim=2,
            layer_norm=True
        )


class FinderCNConfig(FinderConfig):
    """Configuration for the Critical Node (CN) variant"""
    pass


class FinderCNCostConfig(FinderConfig):
    """Configuration for the Critical Node with Cost (CN_cost) variant"""
    MAX_BP_ITER = 2  # Different from the CN variant


class FinderNDConfig(FinderConfig):
    """Configuration for the Network Dismantling (ND) variant"""
    pass


class FinderNDCostConfig(FinderConfig):
    """Configuration for the Network Dismantling with Cost (ND_cost) variant"""
    pass


# Module exports
__all__ = [
    # Message passing layers
    'Structure2VecLayer',
    'GraphSAGELayer',

    # Graph feature encoder
    'GraphFeatureEncoder',
    'GraphFeatureEncoderConfig',

    # FINDER main model
    'FinderGNN',
    'FinderGNNLoss',

    # Configuration classes
    'FinderConfig',
    'FinderCNConfig',
    'FinderCNCostConfig',
    'FinderNDConfig',
    'FinderNDCostConfig',

    # Factory functions
    'create_finder_gnn',
    'create_graph_feature_encoder'
]


# Factory functions for convenient external use
def create_finder_gnn(config: FinderConfig = None, **kwargs) -> FinderGNN:
    """
    Factory function to create a FINDER GNN model

    Args:
        config: FINDER configuration class instance (optional)
        **kwargs: Additional parameters to override the configuration

    Returns:
        Configured FinderGNN instance

    Usage Examples:
    ```python
    # Use default configuration
    model = create_finder_gnn()

    # Use CN variant configuration
    model = create_finder_gnn(FinderCNConfig())

    # Override specific parameters
    model = create_finder_gnn(embedding_size=128, max_bp_iter=5)
    ```
    """
    if config is None:
        config = FinderConfig()

    # Get parameters from configuration
    params = {
        'embedding_size': config.EMBEDDING_SIZE,
        'max_bp_iter': config.MAX_BP_ITER,
        'reg_hidden': config.REG_HIDDEN,
        'aux_dim': config.AUX_DIM,
        'embedding_method': 'graphsage' if config.EMBEDDING_METHOD == 1 else 'structure2vec',
        'aggregator': ['sum', 'mean', 'gcn'][config.AGGREGATOR_ID],
        'initialization_stddev': config.INITIALIZATION_STDDEV
    }

    # Override configuration parameters with kwargs
    params.update(kwargs)

    return FinderGNN(**params)


def create_graph_feature_encoder(config: GraphFeatureEncoderConfig = None, **kwargs) -> GraphFeatureEncoder:
    """
    Factory function to create a graph feature encoder

    Args:
        config: Graph feature encoder configuration (optional)
        **kwargs: Additional parameters to override the configuration

    Returns:
        Configured GraphFeatureEncoder instance

    Usage Examples:
    ```python
    # Use default configuration
    encoder = create_graph_feature_encoder()

    # Use custom configuration
    config = GraphFeatureEncoderConfig(embedding_size=128, max_bp_iter=5)
    encoder = create_graph_feature_encoder(config)

    # Override specific parameters
    encoder = create_graph_feature_encoder(embedding_method='structure2vec')
    ```
    """
    if config is None:
        config = GraphFeatureEncoderConfig()

    # Update configuration with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return GraphFeatureEncoder(config)
