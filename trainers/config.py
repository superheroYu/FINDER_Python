#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Training Configuration System

This module provides configuration management for all four FINDER variants:
- CN (Critical Node): Find critical nodes without cost
- CN_cost (Critical Node with Cost): Find critical nodes with node cost
- ND (Network Dismantling): Dismantle network without cost
- ND_cost (Network Dismantling with Cost): Dismantle network with node cost
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
from enum import Enum
import json


class FinderVariant(Enum):
    """FINDER problem variant"""
    CN = "cn"  # Critical node
    CN_COST = "cn_cost"  # Critical node with cost
    ND = "nd"  # Network dismantling
    ND_COST = "nd_cost"  # Network dismantling with cost


@dataclass
class NetworkConfig:
    """Network generation configuration"""
    min_nodes: int = 30
    max_nodes: int = 50
    # erdos_renyi, powerlaw, small_world, barabasi_albert
    graph_type: str = 'barabasi_albert'
    # uniform, random, degree - node weight assignment strategy
    training_type: str = 'uniform'
    n_train_graphs: int = 1000
    n_valid_graphs: int = 200


@dataclass
class ModelConfig:
    """Neural network model configuration"""
    embedding_size: int = 64
    reg_hidden: int = 32
    aux_dim: int = 4
    max_bp_iter: int = 3
    aggregator_id: int = 0  # 0: sum; 1: mean; 2: GCN
    embedding_method: int = 1  # 0: structure2vec; 1: graphsage
    initialization_stddev: float = 0.01


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Core DQN parameters
    gamma: float = 1.0  # discount factor
    learning_rate: float = 0.0001
    batch_size: int = 64
    memory_size: int = 500000
    n_step: int = 5  # for n-step learning

    # Training schedule
    max_iterations: int = 500000
    update_target_freq: int = 1000
    eval_freq: int = 300
    save_freq: int = 300
    # Graph pool update frequency (aligned with original FINDER)
    graph_pool_update_freq: int = 5000

    # Evaluation during training
    # Whether to evaluate during training (matches original FINDER)
    enable_training_eval: bool = True
    num_eval_episodes: int = 32          # Number of episodes per evaluation
    num_eval_envs: int = 8               # Number of parallel test environments
    # Exploration rate during evaluation (pure greedy policy, matches original FINDER.Test())
    eval_epsilon: float = 0.0

    # Sampling schedule (original FINDER sampling logic)
    sampling_freq: int = 10          # Sample once every N training iterations
    # Number of complete episodes collected per sampling
    episodes_per_sampling: int = 10

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: float = 10000.0

    # Reconstruction loss weight
    alpha: float = 0.001
    max_grad_norm: float = 5.0

    # DQN variants flags
    is_huber_loss: bool = False
    is_double_dqn: bool = False
    is_dueling_dqn: bool = False
    is_prioritized_sampling: bool = False
    is_multi_step_dqn: bool = True

    # Prioritized sampling parameters (if enabled)
    priority_epsilon: float = 0.0000001
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001
    td_err_upper: float = 1.0


@dataclass
class VectorEnvConfig:
    """Vectorized environment configuration"""
    num_envs: int = 8  # Number of parallel environments
    async_env: bool = True  # Use AsyncVectorEnv vs SyncVectorEnv
    max_episode_steps: Optional[int] = None


@dataclass
class FinderTrainingConfig:
    """Complete FINDER training configuration"""
    variant: FinderVariant = FinderVariant.CN
    network: NetworkConfig = field(default_factory=NetworkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vector_env: VectorEnvConfig = field(default_factory=VectorEnvConfig)

    # Paths
    save_dir: str = "./models"
    log_dir: str = "./logs"

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    def __post_init__(self):
        """Post-initialization to create variant-specific configurations"""
        self._apply_variant_specific_configs()

    def _apply_variant_specific_configs(self):
        """Apply variant-specific configuration adjustments"""
        if self.variant == FinderVariant.CN:
            # Critical Node specific settings
            self.model.aux_dim = 4
            self.training.gamma = 1.0
        elif self.variant == FinderVariant.CN_COST:
            # Critical Node with Cost specific settings
            # Consistent with original FINDER: [covered_ratio, edge_covered_ratio, twohop_density, 1.0]
            self.model.aux_dim = 4
            self.training.gamma = 1.0
        elif self.variant == FinderVariant.ND:
            # Network Dismantling specific settings
            self.model.aux_dim = 4
            self.training.gamma = 1.0
        elif self.variant == FinderVariant.ND_COST:
            # Network Dismantling with Cost specific settings
            # Consistent with original FINDER: [covered_ratio, edge_covered_ratio, twohop_density, 1.0]
            self.model.aux_dim = 4
            self.training.gamma = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'variant': self.variant.value,
            'network': self.network.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'vector_env': self.vector_env.__dict__,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
            'device': self.device
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FinderTrainingConfig':
        """Create configuration from dictionary"""
        variant = FinderVariant(config_dict['variant'])

        network_config = NetworkConfig(**config_dict['network'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        vector_env_config = VectorEnvConfig(**config_dict['vector_env'])

        return cls(
            variant=variant,
            network=network_config,
            model=model_config,
            training=training_config,
            vector_env=vector_env_config,
            save_dir=config_dict.get('save_dir', './models'),
            log_dir=config_dict.get('log_dir', './logs'),
            device=config_dict.get('device', 'cuda')
        )

    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FinderTrainingConfig':
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def load_variant_config(variant: FinderVariant) -> FinderTrainingConfig:
    """
    Load configuration from a separate variant configuration file.

    Args:
        variant: FINDER variant

    Returns:
        Training configuration object
    """
    import json

    variant_name = variant.value
    config_file = f"{variant_name}_config.json"
    config_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'configs', config_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # Create configuration object
    network_config = NetworkConfig(
        min_nodes=cfg['network']['min_nodes'],
        max_nodes=cfg['network']['max_nodes'],
        graph_type=cfg['network']['graph_type'],
        training_type=cfg['network']['training_type'],
        n_train_graphs=cfg['network']['n_train_graphs'],
        n_valid_graphs=cfg['network']['n_valid_graphs']
    )

    model_config = ModelConfig(
        embedding_size=cfg['model']['embedding_size'],
        reg_hidden=cfg['model']['reg_hidden'],
        aux_dim=cfg['model']['aux_dim'],
        max_bp_iter=cfg['model']['max_bp_iter'],
        aggregator_id=cfg['model']['aggregator_id'],
        embedding_method=cfg['model']['embedding_method'],
        initialization_stddev=cfg['model']['initialization_stddev']
    )

    training_config = TrainingConfig(
        gamma=cfg['training']['gamma'],
        learning_rate=cfg['training']['learning_rate'],
        batch_size=cfg['training']['batch_size'],
        memory_size=cfg['training']['memory_size'],
        n_step=cfg['training']['n_step'],
        max_iterations=cfg['training']['max_iterations'],
        update_target_freq=cfg['training']['update_target_freq'],
        eval_freq=cfg['training']['eval_freq'],
        save_freq=cfg['training']['save_freq'],
        graph_pool_update_freq=cfg['training']['graph_pool_update_freq'],
        sampling_freq=cfg['training'].get('sampling_freq', 10),
        episodes_per_sampling=cfg['training'].get('episodes_per_sampling', 10),
        eps_start=cfg['training']['eps_start'],
        eps_end=cfg['training']['eps_end'],
        eps_decay_steps=cfg['training']['eps_decay_steps'],
        alpha=cfg['training']['alpha'],
        max_grad_norm=cfg['training'].get('max_grad_norm', 5.0),
        is_huber_loss=cfg['training']['is_huber_loss'],
        is_double_dqn=cfg['training']['is_double_dqn'],
        is_dueling_dqn=cfg['training']['is_dueling_dqn'],
        is_prioritized_sampling=cfg['training']['is_prioritized_sampling'],
        is_multi_step_dqn=cfg['training']['is_multi_step_dqn'],
        priority_epsilon=cfg['training']['priority_epsilon'],
        priority_alpha=cfg['training']['priority_alpha'],
        priority_beta=cfg['training']['priority_beta'],
        priority_beta_increment=cfg['training']['priority_beta_increment'],
        td_err_upper=cfg['training']['td_err_upper'],
        enable_training_eval=cfg['training'].get('enable_training_eval', True),
        num_eval_episodes=cfg['training'].get('num_eval_episodes', 32),
        num_eval_envs=cfg['training'].get('num_eval_envs', 8),
        eval_epsilon=cfg['training'].get('eval_epsilon', 0.0)
    )

    vector_env_config = VectorEnvConfig(
        num_envs=cfg['vector_env']['num_envs'],
        async_env=cfg['vector_env']['async_env'],
        max_episode_steps=cfg['vector_env']['max_episode_steps']
    )

    # Set experiment path
    base_dir = cfg.get('base_dir', './experiments')
    exp_name = cfg.get('experiment_name', f'{variant_name}_experiment')
    exp_dir = os.path.join(base_dir, exp_name)

    return FinderTrainingConfig(
        variant=variant,
        network=network_config,
        model=model_config,
        training=training_config,
        vector_env=vector_env_config,
        save_dir=os.path.join(exp_dir, 'models'),
        log_dir=os.path.join(exp_dir, 'logs'),
        device=cfg.get('device', 'cuda')
    )


def get_default_config(variant: FinderVariant) -> FinderTrainingConfig:
    """
    Get the default configuration for the specified variant.
    Prefer separate per-variant config files, falling back to the unified config file.
    """
    # Prefer loading from separate variant config file
    try:
        return load_variant_config(variant)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(
            f"Warning: Failed to load variant config file ({e}), falling back to unified config")

    # Fall back to the original unified config file approach
    # Prefer loading from configs/finder_defaults.json (if exists)
    try:
        defaults_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'configs', 'finder_defaults.json')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            vkey = variant.value
            base = cfg.get('base', {})
            vcfg = cfg.get('variants', {}).get(vkey, {})

            # Build complete configuration object
            def deep_get(dct, *keys, default=None):
                cur = dct
                for k in keys:
                    if not isinstance(cur, dict) or k not in cur:
                        return default
                    cur = cur[k]
                return cur

            config = FinderTrainingConfig(variant=variant)
            # network
            config.network.min_nodes = deep_get(vcfg, 'network', 'min_nodes', default=deep_get(
                base, 'network', 'min_nodes', default=config.network.min_nodes))
            config.network.max_nodes = deep_get(vcfg, 'network', 'max_nodes', default=deep_get(
                base, 'network', 'max_nodes', default=config.network.max_nodes))
            config.network.graph_type = deep_get(vcfg, 'network', 'graph_type', default=deep_get(
                base, 'network', 'graph_type', default=config.network.graph_type))
            config.network.training_type = deep_get(vcfg, 'network', 'training_type', default=deep_get(
                base, 'network', 'training_type', default=config.network.training_type))
            config.network.n_train_graphs = deep_get(vcfg, 'network', 'n_train_graphs', default=deep_get(
                base, 'network', 'n_train_graphs', default=config.network.n_train_graphs))
            config.network.n_valid_graphs = deep_get(vcfg, 'network', 'n_valid_graphs', default=deep_get(
                base, 'network', 'n_valid_graphs', default=config.network.n_valid_graphs))
            # model
            config.model.embedding_size = deep_get(vcfg, 'model', 'embedding_size', default=deep_get(
                base, 'model', 'embedding_size', default=config.model.embedding_size))
            config.model.reg_hidden = deep_get(vcfg, 'model', 'reg_hidden', default=deep_get(
                base, 'model', 'reg_hidden', default=config.model.reg_hidden))
            config.model.aux_dim = deep_get(vcfg, 'model', 'aux_dim', default=deep_get(
                base, 'model', 'aux_dim', default=config.model.aux_dim))
            config.model.max_bp_iter = deep_get(vcfg, 'model', 'max_bp_iter', default=deep_get(
                base, 'model', 'max_bp_iter', default=config.model.max_bp_iter))
            config.model.aggregator_id = deep_get(vcfg, 'model', 'aggregator_id', default=deep_get(
                base, 'model', 'aggregator_id', default=config.model.aggregator_id))
            config.model.embedding_method = deep_get(vcfg, 'model', 'embedding_method', default=deep_get(
                base, 'model', 'embedding_method', default=config.model.embedding_method))
            config.model.initialization_stddev = deep_get(vcfg, 'model', 'initialization_stddev', default=deep_get(
                base, 'model', 'initialization_stddev', default=config.model.initialization_stddev))
            # training
            config.training.gamma = deep_get(vcfg, 'training', 'gamma', default=deep_get(
                base, 'training', 'gamma', default=config.training.gamma))
            config.training.learning_rate = deep_get(vcfg, 'training', 'learning_rate', default=deep_get(
                base, 'training', 'learning_rate', default=config.training.learning_rate))
            config.training.batch_size = deep_get(vcfg, 'training', 'batch_size', default=deep_get(
                base, 'training', 'batch_size', default=config.training.batch_size))
            config.training.memory_size = deep_get(vcfg, 'training', 'memory_size', default=deep_get(
                base, 'training', 'memory_size', default=config.training.memory_size))
            config.training.n_step = deep_get(vcfg, 'training', 'n_step', default=deep_get(
                base, 'training', 'n_step', default=config.training.n_step))
            config.training.max_iterations = deep_get(vcfg, 'training', 'max_iterations', default=deep_get(
                base, 'training', 'max_iterations', default=config.training.max_iterations))
            config.training.update_target_freq = deep_get(vcfg, 'training', 'update_target_freq', default=deep_get(
                base, 'training', 'update_target_freq', default=config.training.update_target_freq))
            config.training.eval_freq = deep_get(vcfg, 'training', 'eval_freq', default=deep_get(
                base, 'training', 'eval_freq', default=config.training.eval_freq))
            config.training.save_freq = deep_get(vcfg, 'training', 'save_freq', default=deep_get(
                base, 'training', 'save_freq', default=config.training.save_freq))
            config.training.graph_pool_update_freq = deep_get(vcfg, 'training', 'graph_pool_update_freq', default=deep_get(
                base, 'training', 'graph_pool_update_freq', default=config.training.graph_pool_update_freq))
            config.training.sampling_freq = deep_get(vcfg, 'training', 'sampling_freq', default=deep_get(
                base, 'training', 'sampling_freq', default=config.training.sampling_freq))
            config.training.episodes_per_sampling = deep_get(vcfg, 'training', 'episodes_per_sampling', default=deep_get(
                base, 'training', 'episodes_per_sampling', default=config.training.episodes_per_sampling))
            config.training.eps_start = deep_get(vcfg, 'training', 'eps_start', default=deep_get(
                base, 'training', 'eps_start', default=config.training.eps_start))
            config.training.eps_end = deep_get(vcfg, 'training', 'eps_end', default=deep_get(
                base, 'training', 'eps_end', default=config.training.eps_end))
            config.training.eps_decay_steps = deep_get(vcfg, 'training', 'eps_decay_steps', default=deep_get(
                base, 'training', 'eps_decay_steps', default=config.training.eps_decay_steps))
            config.training.alpha = deep_get(vcfg, 'training', 'alpha', default=deep_get(
                base, 'training', 'alpha', default=config.training.alpha))
            config.training.max_grad_norm = deep_get(vcfg, 'training', 'max_grad_norm', default=deep_get(
                base, 'training', 'max_grad_norm', default=config.training.max_grad_norm))
            config.training.is_huber_loss = deep_get(vcfg, 'training', 'is_huber_loss', default=deep_get(
                base, 'training', 'is_huber_loss', default=config.training.is_huber_loss))
            config.training.is_double_dqn = deep_get(vcfg, 'training', 'is_double_dqn', default=deep_get(
                base, 'training', 'is_double_dqn', default=config.training.is_double_dqn))
            config.training.is_dueling_dqn = deep_get(vcfg, 'training', 'is_dueling_dqn', default=deep_get(
                base, 'training', 'is_dueling_dqn', default=config.training.is_dueling_dqn))
            config.training.is_prioritized_sampling = deep_get(vcfg, 'training', 'is_prioritized_sampling', default=deep_get(
                base, 'training', 'is_prioritized_sampling', default=config.training.is_prioritized_sampling))
            config.training.is_multi_step_dqn = deep_get(vcfg, 'training', 'is_multi_step_dqn', default=deep_get(
                base, 'training', 'is_multi_step_dqn', default=config.training.is_multi_step_dqn))
            config.training.priority_epsilon = deep_get(vcfg, 'training', 'priority_epsilon', default=deep_get(
                base, 'training', 'priority_epsilon', default=config.training.priority_epsilon))
            config.training.priority_alpha = deep_get(vcfg, 'training', 'priority_alpha', default=deep_get(
                base, 'training', 'priority_alpha', default=config.training.priority_alpha))
            config.training.priority_beta = deep_get(vcfg, 'training', 'priority_beta', default=deep_get(
                base, 'training', 'priority_beta', default=config.training.priority_beta))
            config.training.priority_beta_increment = deep_get(vcfg, 'training', 'priority_beta_increment', default=deep_get(
                base, 'training', 'priority_beta_increment', default=config.training.priority_beta_increment))
            config.training.td_err_upper = deep_get(vcfg, 'training', 'td_err_upper', default=deep_get(
                base, 'training', 'td_err_upper', default=config.training.td_err_upper))
            config.training.enable_training_eval = deep_get(vcfg, 'training', 'enable_training_eval', default=deep_get(
                base, 'training', 'enable_training_eval', default=config.training.enable_training_eval))
            config.training.num_eval_episodes = deep_get(vcfg, 'training', 'num_eval_episodes', default=deep_get(
                base, 'training', 'num_eval_episodes', default=config.training.num_eval_episodes))
            config.training.num_eval_envs = deep_get(vcfg, 'training', 'num_eval_envs', default=deep_get(
                base, 'training', 'num_eval_envs', default=config.training.num_eval_envs))
            config.training.eval_epsilon = deep_get(vcfg, 'training', 'eval_epsilon', default=deep_get(
                base, 'training', 'eval_epsilon', default=config.training.eval_epsilon))

            # vector env
            config.vector_env.num_envs = deep_get(vcfg, 'vector_env', 'num_envs', default=deep_get(
                base, 'vector_env', 'num_envs', default=config.vector_env.num_envs))
            config.vector_env.async_env = deep_get(vcfg, 'vector_env', 'async_env', default=deep_get(
                base, 'vector_env', 'async_env', default=config.vector_env.async_env))
            config.vector_env.max_episode_steps = deep_get(vcfg, 'vector_env', 'max_episode_steps', default=deep_get(
                base, 'vector_env', 'max_episode_steps', default=config.vector_env.max_episode_steps))

            # Device and path
            config.device = cfg.get('device', config.device)
            base_dir = cfg.get('base_dir', './experiments')
            exp_name = cfg.get('default_experiment_name', f'{vkey}_exp')
            exp_dir = os.path.join(base_dir, exp_name)
            config.save_dir = os.path.join(exp_dir, 'models')
            config.log_dir = os.path.join(exp_dir, 'logs')

            # Apply variant-specific adjustments
            config._apply_variant_specific_configs()
            return config
    except Exception:
        # Fall back to hardcoded defaults
        pass
    return FinderTrainingConfig(variant=variant)


def create_experiment_config(
    variant: FinderVariant,
    experiment_name: str,
    base_dir: str = "./experiments"
) -> FinderTrainingConfig:
    """Create configuration for a specific experiment"""
    config = get_default_config(variant)

    # Set experiment-specific paths
    exp_dir = os.path.join(base_dir, experiment_name)
    config.save_dir = os.path.join(exp_dir, "models")
    config.log_dir = os.path.join(exp_dir, "logs")

    return config


# Predefined configurations for common experiments
FAST_DEBUG_CONFIG = {
    'max_iterations': 1000,
    'n_train_graphs': 100,
    'n_valid_graphs': 20,
    'eval_freq': 100,
    'save_freq': 500,
    'memory_size': 10000,
    'num_envs': 2
}

PRODUCTION_CONFIG = {
    'max_iterations': 500000,
    'n_train_graphs': 1000,
    'n_valid_graphs': 200,
    'eval_freq': 300,
    'save_freq': 300,
    'memory_size': 500000,
    'num_envs': 8
}


def apply_config_preset(config: FinderTrainingConfig, preset_name: str) -> FinderTrainingConfig:
    """Apply a configuration preset to modify the config"""
    presets = {
        'fast_debug': FAST_DEBUG_CONFIG,
        'production': PRODUCTION_CONFIG
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    preset = presets[preset_name]

    # Apply preset values
    for key, value in preset.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.network, key):
            setattr(config.network, key, value)
        elif hasattr(config.vector_env, key):
            setattr(config.vector_env, key, value)

    return config
