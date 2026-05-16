#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Training Infrastructure Package

This package provides a complete training infrastructure for the FINDER deep
reinforcement learning framework, adapted for pure Python implementation.

Main Components:
- config: Configuration management for all FINDER variants
- vector_trainer: Main trainer with vectorized environment support
- replay_buffer: Experience replay implementations (standard and prioritized)
- models.data_interfaces: NetworkX observation to PyG graph batch conversion
- utils: Logging, error handling, and training utilities
"""

from .config import (
    FinderTrainingConfig,
    FinderVariant,
    NetworkConfig,
    ModelConfig,
    TrainingConfig,
    VectorEnvConfig,
    get_default_config,
    create_experiment_config,
    apply_config_preset
)

from .vector_trainer import (
    FinderVectorTrainer,
    create_trainer,
    run_training_experiment
)

# vector_env removed; for gym vector environments, use envs.gym_batch.make_gym_batch_env

from .replay_buffer import (
    Experience,
    ReplaySample,
    NStepReplayBuffer,
    PrioritizedReplayBuffer,
    create_replay_buffer
)

# data_pipeline removed (graph data batching now provided by models.data_interfaces)

from .utils import (
    TrainingMetrics,
    FinderLogger,
    PerformanceMonitor,
    CheckpointManager,
    HyperparameterValidator,
    MetricsAnalyzer,
    setup_training_environment,
    save_training_artifacts,
    load_training_artifacts
)

__version__ = "1.0.0"
__author__ = "FINDER Python contributors"

# Package-level configuration
DEFAULT_DEVICE = "cuda"
DEFAULT_LOG_LEVEL = "INFO"

# Supported FINDER variants
SUPPORTED_VARIANTS = [
    FinderVariant.CN,
    FinderVariant.CN_COST,
    FinderVariant.ND,
    FinderVariant.ND_COST
]

def get_package_info() -> dict:
    """Get package information"""
    return {
        "name": "FINDER Training Infrastructure",
        "version": __version__,
        "author": __author__,
        "supported_variants": [v.value for v in SUPPORTED_VARIANTS],
        "components": [
            "Configuration Management",
            "Vectorized Training",
            "Experience Replay",
            "Graph Data Interfaces",
            "Logging & Monitoring"
        ]
    }

def quick_start_training(
    variant: str = "cn",
    preset: str = "fast_debug",
    experiment_name: str = "quickstart"
) -> dict:
    """
    Quick start training with minimal configuration.
    
    Args:
        variant: FINDER variant ('cn', 'cn_cost', 'nd', 'nd_cost')
        preset: Configuration preset ('fast_debug', 'production')
        experiment_name: Name for the experiment
        
    Returns:
        Training results
    """
    # Validate variant
    variant_enum = FinderVariant(variant)

    # Create configuration and apply the requested experiment path/preset.
    config = create_experiment_config(variant_enum, experiment_name)
    config = apply_config_preset(config, preset)

    trainer = FinderVectorTrainer(config)
    try:
        return trainer.train()
    finally:
        trainer.cleanup()

# Export all main components
__all__ = [
    # Configuration
    'FinderTrainingConfig',
    'FinderVariant',
    'NetworkConfig',
    'ModelConfig', 
    'TrainingConfig',
    'VectorEnvConfig',
    'get_default_config',
    'create_experiment_config',
    'apply_config_preset',
    
    # Trainer
    'FinderVectorTrainer',
    'create_trainer',
    'run_training_experiment',
    
    # Environment
    # Use envs.gym_batch.make_gym_batch_env as the vector environment factory
    
    # Replay Buffer
    'Experience',
    'ReplaySample',
    'NStepReplayBuffer',
    'PrioritizedReplayBuffer',
    'create_replay_buffer',
    
    # Utils
    'TrainingMetrics',
    'FinderLogger',
    'PerformanceMonitor',
    'CheckpointManager',
    'HyperparameterValidator',
    'MetricsAnalyzer',
    'setup_training_environment',
    'save_training_artifacts',
    'load_training_artifacts',
    
    # Convenience functions
    'get_package_info',
    'quick_start_training',
    
    # Constants
    'SUPPORTED_VARIANTS',
    'DEFAULT_DEVICE',
    'DEFAULT_LOG_LEVEL'
]
