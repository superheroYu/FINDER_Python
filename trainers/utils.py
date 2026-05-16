#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Training Utilities

This module provides utility functions for logging, error handling,
hyperparameter management, and training diagnostics.

Features:
- Comprehensive logging system with file and console output
- Error handling and exception management
- Training metrics collection and analysis
- Hyperparameter validation and serialization
- Checkpoint and model persistence utilities
- Performance monitoring and profiling
"""

import os
import sys
import time
import json
import pickle
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    # Loss metrics
    total_loss: List[float]
    q_loss: List[float] 
    reconstruction_loss: List[float]
    
    # Performance metrics
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rate: List[float]
    
    # Training dynamics
    exploration_rate: List[float]
    learning_rate: List[float]
    replay_buffer_size: List[int]
    
    # Model metrics
    q_value_mean: List[float]
    q_value_std: List[float]
    gradient_norm: List[float]
    
    # System metrics
    memory_usage: List[float]
    gpu_utilization: List[float]
    training_time: List[float]
    
    def __post_init__(self):
        """Initialize empty lists if not provided"""
        for field_name, field_type in self.__annotations__.items():
            if not hasattr(self, field_name):
                setattr(self, field_name, [])


class FinderLogger:
    """
    Enhanced logging system for FINDER training.
    
    Features:
    - Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File and console output
    - Tensorboard integration
    - Performance metrics logging
    - Exception tracking
    """
    
    def __init__(
        self,
        name: str = 'FinderTrainer',
        log_dir: str = './logs',
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        tensorboard: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Minimum logging level
            console_output: Enable console logging
            file_output: Enable file logging
            tensorboard: Enable tensorboard logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            log_file = self.log_dir / f'{name.lower()}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Tensorboard writer
        self.writer = None
        if tensorboard:
            tb_dir = self.log_dir / 'tensorboard'
            self.writer = SummaryWriter(tb_dir)
        
        # Error tracking
        self.error_count = 0
        self.error_log = []
        
        self.logger.info(f"Initialized {name} logger")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message)
        if self.writer and kwargs:
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Info/{key}', value, kwargs.get('step', 0))
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message with optional exception"""
        self.error_count += 1
        
        if exception:
            error_info = {
                'message': message,
                'exception': str(exception),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            self.error_log.append(error_info)
            self.logger.error(f"{message}: {exception}")
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.error(message)
    
    def critical(self, message: str, exception: Optional[Exception] = None):
        """Log critical message"""
        if exception:
            self.logger.critical(f"{message}: {exception}")
            self.logger.debug(traceback.format_exc())
        else:
            self.logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to tensorboard"""
        if not self.writer:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                tag = f'{prefix}/{key}' if prefix else key
                self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram to tensorboard"""
        if self.writer:
            self.writer.add_histogram(name, values, step)
    
    def log_model_graph(self, model: nn.Module, input_data: torch.Tensor):
        """Log model graph to tensorboard"""
        if self.writer:
            try:
                self.writer.add_graph(model, input_data)
            except Exception as e:
                self.warning(f"Failed to log model graph: {e}")
    
    def close(self):
        """Close logger and cleanup"""
        if self.writer:
            self.writer.close()
        
        # Log error summary
        if self.error_log:
            error_file = self.log_dir / 'errors.json'
            with open(error_file, 'w') as f:
                json.dump(self.error_log, f, indent=2)
            self.logger.info(f"Logged {len(self.error_log)} errors to {error_file}")


class PerformanceMonitor:
    """Monitor system performance during training"""
    
    def __init__(self, logger: Optional[FinderLogger] = None):
        """Initialize performance monitor"""
        self.logger = logger
        self.metrics = {
            'cpu_percent': deque(maxlen=1000),
            'memory_percent': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'gpu_utilization': deque(maxlen=1000)
        }
        self.start_time = time.time()
    
    def update(self):
        """Update performance metrics"""
        # CPU and system memory
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.metrics['gpu_memory'].append(gpu_memory * 100)
                # GPU utilization would require nvidia-ml-py package
                self.metrics['gpu_utilization'].append(0.0)  # Placeholder
            except Exception:
                self.metrics['gpu_memory'].append(0.0)
                self.metrics['gpu_utilization'].append(0.0)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'cpu_percent': self.metrics['cpu_percent'][-1] if self.metrics['cpu_percent'] else 0.0,
            'memory_percent': self.metrics['memory_percent'][-1] if self.metrics['memory_percent'] else 0.0,
            'gpu_memory': self.metrics['gpu_memory'][-1] if self.metrics['gpu_memory'] else 0.0,
            'training_time': time.time() - self.start_time
        }
    
    def get_average_metrics(self, last_n: int = 100) -> Dict[str, float]:
        """Get average metrics over last n samples"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                recent_values = list(values)[-last_n:]
                avg_metrics[f'avg_{key}'] = float(np.mean(recent_values))
        return avg_metrics


class CheckpointManager:
    """Manage model checkpoints and training state"""
    
    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 10,
        logger: Optional[FinderLogger] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            logger: Logger instance
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logger
        
        # Track saved checkpoints
        self.checkpoints = []
        self._load_checkpoint_list()
    
    def _load_checkpoint_list(self):
        """Load list of existing checkpoints"""
        checkpoint_pattern = self.save_dir.glob('checkpoint_*.pt')
        self.checkpoints = sorted(list(checkpoint_pattern))
        
        # Keep only the latest checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            for old_checkpoint in self.checkpoints[:-self.max_checkpoints]:
                try:
                    old_checkpoint.unlink()
                    if self.logger:
                        self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
            
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        iteration: int,
        is_best: bool = False,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save training checkpoint.
        
        Args:
            state_dict: Model and training state
            iteration: Current training iteration
            is_best: Whether this is the best model so far
            extra_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'checkpoint_{iteration:06d}_{timestamp}.pt'
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': time.time(),
            'state_dict': state_dict,
            'is_best': is_best
        }
        
        if extra_info:
            checkpoint_data.update(extra_info)
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.checkpoints.append(checkpoint_path)
            
            if self.logger:
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model separately
            if is_best:
                best_path = self.save_dir / 'best_model.pt'
                torch.save(checkpoint_data, best_path)
                if self.logger:
                    self.logger.info(f"Saved best model: {best_path}")
            
            # Clean up old checkpoints
            self._load_checkpoint_list()
            
            return checkpoint_path
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Checkpoint data
        """
        if checkpoint_path is None:
            if not self.checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = self.checkpoints[-1]
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            if self.logger:
                self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        return self.checkpoints[-1] if self.checkpoints else None


class HyperparameterValidator:
    """Validate and manage hyperparameters"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Training parameters
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not (0 < lr < 1):
                errors.append(f"Learning rate must be in (0, 1), got {lr}")
        
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"Batch size must be positive integer, got {batch_size}")
        
        if 'gamma' in config:
            gamma = config['gamma']
            if not (0 <= gamma <= 1):
                errors.append(f"Gamma must be in [0, 1], got {gamma}")
        
        # Memory parameters
        if 'memory_size' in config:
            memory_size = config['memory_size']
            if not isinstance(memory_size, int) or memory_size <= 0:
                errors.append(f"Memory size must be positive integer, got {memory_size}")
        
        # Network parameters
        if 'embedding_size' in config:
            embedding_size = config['embedding_size']
            if not isinstance(embedding_size, int) or embedding_size <= 0:
                errors.append(f"Embedding size must be positive integer, got {embedding_size}")
        
        # Environment parameters
        if 'max_nodes' in config:
            max_nodes = config['max_nodes']
            if not isinstance(max_nodes, int) or max_nodes <= 0:
                errors.append(f"Max nodes must be positive integer, got {max_nodes}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def suggest_hyperparameters(variant: str) -> Dict[str, Any]:
        """Suggest hyperparameters for a FINDER variant"""
        base_params = {
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 1.0,
            'memory_size': 500000,
            'embedding_size': 64,
            'n_step': 5
        }
        
        variant_specific = {
            'cn': {'gamma': 1.0, 'aux_dim': 4},
            'cn_cost': {'gamma': 1.0, 'aux_dim': 4},  # Consistent with original FINDER
            'nd': {'gamma': 1.0, 'aux_dim': 4},
            'nd_cost': {'gamma': 1.0, 'aux_dim': 4}   # Consistent with original FINDER
        }
        
        if variant in variant_specific:
            base_params.update(variant_specific[variant])
        
        return base_params


class MetricsAnalyzer:
    """Analyze and visualize training metrics"""
    
    def __init__(self, logger: Optional[FinderLogger] = None):
        self.logger = logger
    
    def analyze_convergence(self, losses: List[float], window_size: int = 100) -> Dict[str, float]:
        """Analyze loss convergence"""
        if len(losses) < window_size:
            return {'convergence_rate': 0.0, 'is_converging': False}
        
        # Calculate moving average
        losses = np.array(losses)
        moving_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        
        # Calculate convergence rate (slope of recent trend)
        recent_trend = moving_avg[-window_size:]
        if len(recent_trend) > 1:
            x = np.arange(len(recent_trend))
            convergence_rate = np.polyfit(x, recent_trend, 1)[0]
            is_converging = convergence_rate < -1e-6  # Negative slope indicates convergence
        else:
            convergence_rate = 0.0
            is_converging = False
        
        return {
            'convergence_rate': float(convergence_rate),
            'is_converging': is_converging,
            'recent_loss': float(moving_avg[-1]) if len(moving_avg) > 0 else 0.0
        }
    
    def analyze_exploration(self, rewards: List[float], exploration_rates: List[float]) -> Dict[str, float]:
        """Analyze exploration vs exploitation balance"""
        if len(rewards) < 10 or len(exploration_rates) < 10:
            return {'exploration_efficiency': 0.0}
        
        rewards = np.array(rewards)
        exploration_rates = np.array(exploration_rates)
        
        # Calculate correlation between exploration rate and reward improvement
        if len(rewards) > 1:
            reward_deltas = np.diff(rewards)
            if len(reward_deltas) == len(exploration_rates) - 1:
                correlation = np.corrcoef(exploration_rates[1:], reward_deltas)[0, 1]
                exploration_efficiency = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                exploration_efficiency = 0.0
        else:
            exploration_efficiency = 0.0
        
        return {
            'exploration_efficiency': float(exploration_efficiency),
            'current_exploration_rate': float(exploration_rates[-1]),
            'average_reward': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
        }
    
    def generate_training_report(self, metrics: TrainingMetrics, save_path: Optional[str] = None) -> str:
        """Generate comprehensive training report"""
        report_lines = [
            "FINDER Training Report",
            "=" * 50,
            "",
            f"Training Episodes: {len(metrics.episode_rewards)}",
            f"Total Training Steps: {len(metrics.total_loss)}",
            ""
        ]
        
        if metrics.episode_rewards:
            report_lines.extend([
                "Episode Performance:",
                f"  Average Reward: {np.mean(metrics.episode_rewards):.4f}",
                f"  Best Reward: {np.max(metrics.episode_rewards):.4f}",
                f"  Reward Std: {np.std(metrics.episode_rewards):.4f}",
                ""
            ])
        
        if metrics.total_loss:
            convergence_analysis = self.analyze_convergence(metrics.total_loss)
            report_lines.extend([
                "Loss Analysis:",
                f"  Final Loss: {metrics.total_loss[-1]:.6f}",
                f"  Convergence Rate: {convergence_analysis['convergence_rate']:.8f}",
                f"  Is Converging: {convergence_analysis['is_converging']}",
                ""
            ])
        
        if metrics.episode_rewards and metrics.exploration_rate:
            exploration_analysis = self.analyze_exploration(metrics.episode_rewards, metrics.exploration_rate)
            report_lines.extend([
                "Exploration Analysis:",
                f"  Exploration Efficiency: {exploration_analysis['exploration_efficiency']:.4f}",
                f"  Final Exploration Rate: {exploration_analysis['current_exploration_rate']:.4f}",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            if self.logger:
                self.logger.info(f"Training report saved to: {save_path}")
        
        return report


# Utility functions

def setup_training_environment(config: Dict[str, Any], log_dir: str) -> Tuple[FinderLogger, PerformanceMonitor, CheckpointManager]:
    """Set up complete training environment with logging and monitoring"""
    # Validate configuration
    is_valid, errors = HyperparameterValidator.validate_config(config)
    if not is_valid:
        raise ValueError(f"Invalid configuration: {errors}")
    
    # Create logger
    logger = FinderLogger(
        name='FinderTrainer',
        log_dir=log_dir,
        log_level=logging.INFO,
        tensorboard=True
    )
    
    # Create performance monitor
    monitor = PerformanceMonitor(logger)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=os.path.join(log_dir, 'checkpoints'),
        logger=logger
    )
    
    logger.info("Training environment setup complete")
    return logger, monitor, checkpoint_manager


def save_training_artifacts(
    metrics: TrainingMetrics,
    config: Dict[str, Any],
    save_dir: str,
    logger: Optional[FinderLogger] = None
):
    """Save all training artifacts"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = save_path / 'training_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save configuration
    config_file = save_path / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate and save report
    analyzer = MetricsAnalyzer(logger)
    report_file = save_path / 'training_report.txt'
    analyzer.generate_training_report(metrics, str(report_file))
    
    if logger:
        logger.info(f"Training artifacts saved to: {save_dir}")


def load_training_artifacts(load_dir: str) -> Tuple[TrainingMetrics, Dict[str, Any]]:
    """Load training artifacts"""
    load_path = Path(load_dir)
    
    # Load metrics
    metrics_file = load_path / 'training_metrics.pkl'
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    # Load configuration
    config_file = load_path / 'config.json'  
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return metrics, config
