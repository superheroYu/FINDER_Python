"""
FINDER Neural Network Models - PyTorch Implementation

This package contains PyTorch implementations of the FINDER neural network
architecture, converted from the original Cython/TensorFlow implementation.

Main Components:
- gnn_arch.py: Core graph neural network architecture
- policy_net.py: Policy networks for reinforcement learning
- data_interfaces.py: Data format specifications for trainer/model integration

Usage:
    from models import FinderGNN, FinderPolicyNetwork, create_policy_network
    from models.data_interfaces import GraphBatch, create_graph_batch_from_observations
"""

from .gnn_arch import (
    FinderGNN,
    FinderGNNLoss,
    Structure2VecLayer,
    GraphSAGELayer,
    FinderConfig,
    FinderCNConfig,
    FinderCNCostConfig,
    FinderNDConfig,
    FinderNDCostConfig
)

from .policy_net import (
    FinderPolicyNetwork,
    DoubleDQNPolicyNetwork,
    DuelingDQNPolicyNetwork,
    create_policy_network
)

__all__ = [
    'FinderGNN',
    'FinderGNNLoss',
    'Structure2VecLayer', 
    'GraphSAGELayer',
    'FinderConfig',
    'FinderCNConfig',
    'FinderCNCostConfig',
    'FinderNDConfig',
    'FinderNDCostConfig',
    'FinderPolicyNetwork',
    'DoubleDQNPolicyNetwork',
    'DuelingDQNPolicyNetwork',
    'create_policy_network'
]
