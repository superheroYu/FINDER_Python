"""
FINDER Environment Package

This package provides Python implementations of FINDER deep reinforcement learning
environments for graph-based optimization problems.

Contains four environment variants:
- Critical Node (CN): Minimize connected component decomposition score
- Critical Node with Cost (CN_cost): CN with node removal costs
- Network Dismantling (ND): Minimize the largest connected component size
- Network Dismantling with Cost (ND_cost): ND with node removal costs

All environments follow the gymnasium interface, compatible with standard RL frameworks.

Usage:
    from envs import CriticalNodeEnv, NetworkDismantlingEnv

    # Create environment
    env = CriticalNodeEnv(max_nodes=50)

    # Reset and step
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from .base_env import BaseFINDEREnv
from .cn_env import CriticalNodeEnv
from .cn_cost_env import CriticalNodeCostEnv
from .nd_env import NetworkDismantlingEnv
from .nd_cost_env import NetworkDismantlingCostEnv

# Version info
__version__ = "1.0.0"
__author__ = "FINDER Python Conversion Team"

# Environment registry for easy access (single environments)
ENVIRONMENT_REGISTRY = {
    'cn': CriticalNodeEnv,
    'critical_node': CriticalNodeEnv,
    'cn_cost': CriticalNodeCostEnv,
    'critical_node_cost': CriticalNodeCostEnv,
    'nd': NetworkDismantlingEnv,
    'network_dismantling': NetworkDismantlingEnv,
    'nd_cost': NetworkDismantlingCostEnv,
    'network_dismantling_cost': NetworkDismantlingCostEnv,
}

# Export all public classes
__all__ = [
    'BaseFINDEREnv',
    'CriticalNodeEnv',
    'CriticalNodeCostEnv',
    'NetworkDismantlingEnv',
    'NetworkDismantlingCostEnv',
    'make_env',
    'make_batch_env',
    'list_environments'
]


def make_env(env_type: str, config=None, **kwargs):
    """
    Factory function to create an environment by name.

    Args:
        env_type: Environment type name (e.g. 'cn', 'nd', 'cn_cost', 'nd_cost')
        config: Training config object for variant-specific parameters
        **kwargs: Arguments passed to the environment constructor

    Returns:
        Environment instance

    Examples:
        env = make_env('cn', max_nodes=50, seed=42)
        env = make_env('cn_cost', config=training_config)
    """
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment type: {env_type}. "
                        f"Available types: {list(ENVIRONMENT_REGISTRY.keys())}")

    # If config provided, extract relevant parameters from it
    if config is not None:
        # Extract environment parameters from config
        if hasattr(config, 'network'):
            kwargs.setdefault('max_nodes', config.network.max_nodes)
            kwargs.setdefault('min_nodes', config.network.min_nodes)
            kwargs.setdefault('graph_type', config.network.graph_type)
            # Check for training_type
            if hasattr(config.network, 'training_type'):
                kwargs.setdefault('training_type', config.network.training_type)

        if hasattr(config, 'model'):
            kwargs.setdefault('aux_dim', config.model.aux_dim)

    env_class = ENVIRONMENT_REGISTRY[env_type]
    return env_class(**kwargs)


def make_batch_env(env_type: str, batch_size: int = 64, **kwargs):
    """
    Create a batch environment using gymnasium.vector (zero-copy observations).
    Compatible with historical interface, internally delegates to envs.gym_batch.make_gym_batch_env.
    """
    from .gym_batch import make_gym_batch_env
    return make_gym_batch_env(env_type, batch_size=batch_size, **kwargs)


def list_environments():
    """
    List all available environment types.

    Returns:
        Dictionary mapping environment names to their classes
    """
    return ENVIRONMENT_REGISTRY.copy()


# Problem type mapping for backward compatibility
PROBLEM_TYPES = {
    'critical_node': ['cn', 'cn_cost'],
    'network_dismantling': ['nd', 'nd_cost'],
    'unweighted': ['cn', 'nd'],
    'weighted': ['cn_cost', 'nd_cost']
}


def get_environments_by_problem_type(problem_type: str):
    """
    Get environment types by problem category.

    Args:
        problem_type: One of 'critical_node', 'network_dismantling', 'unweighted', 'weighted'

    Returns:
        List of environment type names
    """
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"Unknown problem type: {problem_type}. "
                        f"Available types: {list(PROBLEM_TYPES.keys())}")

    return PROBLEM_TYPES[problem_type]
