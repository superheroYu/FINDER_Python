#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Vectorized Trainer

This module implements the main training infrastructure for FINDER,
using vectorized environments for parallel sampling. Integrates:

- VectorEnv for parallel environment execution
- Experience replay buffers (standard and prioritized)
- Training loop compatible with the original FINDER methodology
- Data pipeline for FINDER environments and policy networks

Main features:
- Supports all four FINDER variants (CN, CN_cost, ND, ND_cost)
- Parallel environment sampling for improved efficiency
- N-step learning and prioritized experience replay
- Integration with PyTorch and PyTorch Geometric models
- Comprehensive logging and checkpointing
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# Config and replay
from .config import FinderTrainingConfig, FinderVariant
from .replay_buffer import PrioritizedReplayBuffer, create_replay_buffer

# Use gym vector environment adapter (keep observations as dict lists)
from envs.gym_batch import make_gym_batch_env
from envs.graph_pool import regenerate_all_pools, get_all_pool_stats

# Policy model and data interface
from models.policy_net import FinderPolicyNetwork, create_policy_network
from models.gnn_arch import FinderConfig
from models.data_interfaces import (
    create_graph_batch_from_observations,
)


class FinderVectorTrainer:
    """
    Main FINDER trainer class using vectorized environments.

    This class unifies the entire training process:
    1. Creates vectorized environments for parallel sampling
    2. Manages experience replay buffers
    3. Coordinates with the policy network
    4. Implements the DQN training loop with FINDER-specific adaptations
    5. Handles logging, checkpointing, and evaluation
    """

    def __init__(
        self,
        config: FinderTrainingConfig,
        env_class: Optional[type] = None,
        model_class: Optional[type] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the vectorized trainer.

        Args:
            config: Training configuration
            env_class: Optional custom environment class
            model_class: Optional custom model class
            logger: External logger (creates its own if None)
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')
        self.base_seed = int(os.environ.get('FINDER_SEED', '42'))

        # Set up logging
        self.logger = logger or self._setup_logging()

        # Print detailed configuration summary
        self._print_config_summary()

        # Initialize environment using the registered batch environment factory
        # Ignored when an external concrete class is passed; use registered factory
        self.env_class = env_class
        self.vec_env = self._create_vectorized_environment()
        try:
            self.vec_env.reset(seed=self.base_seed)
        except Exception as e:
            self.logger.warning(f"Failed to seed training environment: {e}")

        # Create dedicated test environment (for training-time metric recording; only created when enabled)
        self.test_vec_env = None
        if config.training.enable_training_eval:
            self.test_vec_env = self._create_test_environment(
                config.training.num_eval_envs)
            try:
                self.test_vec_env.reset(seed=self.base_seed + 100000)
            except Exception as e:
                self.logger.warning(f"Failed to seed test environment: {e}")

        # Initialize model
        self.model_class = model_class
        self.q_network: Optional[FinderPolicyNetwork] = None
        # Kept only for old checkpoint compatibility
        self.target_network: Optional[FinderPolicyNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Initialize replay buffer
        self.replay_buffer = self._create_replay_buffer()

        # Training state
        self.global_step = 0
        # Current training iteration (used for epsilon decay)
        self.current_iteration = 0
        self.episode_count = 0
        self.best_performance = float('-inf')

        # Stats tracking
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'loss_history': [],
            'exploration_rate': [],
            'replay_buffer_size': []
        }

        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(config.log_dir)

        self.logger.info(
            f"Initialized FINDER trainer for variant: {config.variant.value}")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Vector environments: {config.vector_env.num_envs}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('FinderTrainer')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_file = os.path.join(self.config.log_dir, 'training.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _print_config_summary(self):
        """Print detailed configuration summary."""
        config = self.config
        print("=" * 60)
        print(
            f"FINDER {config.variant.value.upper()} Variant Training Configuration")
        print("=" * 60)

        # Network config
        print("Network Configuration:")
        print(f"  Graph type: {config.network.graph_type}")
        print(f"  Weight strategy: {config.network.training_type}")
        print(
            f"  Graph node range: {config.network.min_nodes}-{config.network.max_nodes}")

        # Model config
        print("\nModel Configuration:")
        print(f"  Embedding dim: {config.model.embedding_size}")
        print(f"  Conv layers: {config.model.max_bp_iter}")
        print(f"  Hidden size: {config.model.reg_hidden}")
        print(f"  Aux feature dim: {config.model.aux_dim}")

        # Training config
        print("\nTraining Configuration:")
        print(f"  Training steps: {config.training.max_iterations:,}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Replay buffer: {config.training.memory_size:,}")
        print(f"  N-step learning: {config.training.n_step}-step")
        print(f"  Reconstruction loss weight: {config.training.alpha}")
        print(
            f"  Target network update: every {config.training.update_target_freq} steps")

        # DQN variant config
        print("\nDQN Configuration:")
        # Determine DQN type description
        if config.training.is_dueling_dqn:
            if config.training.is_double_dqn:
                dqn_type = "Dueling Double DQN"
            else:
                dqn_type = "Dueling DQN (inherits Double DQN)"
        elif config.training.is_double_dqn:
            dqn_type = "Double DQN"
        else:
            dqn_type = "Vanilla DQN"
        print(f"  DQN type: {dqn_type}")
        print(
            f"  Loss function: {'Huber Loss' if config.training.is_huber_loss else 'MSE Loss'}")
        print(
            f"  Prioritized sampling: {'Enabled' if config.training.is_prioritized_sampling else 'Disabled'}")
        print(
            f"  Multi-step learning: {'Enabled' if config.training.is_multi_step_dqn else 'Disabled'}")

        # Exploration config
        print(
            f"  epsilon-greedy: {config.training.eps_start} -> {config.training.eps_end}")
        print(f"  epsilon decay steps: {config.training.eps_decay_steps:.0f}")

        # Sampling config
        print(
            f"  Sampling frequency: every {config.training.sampling_freq} training iterations")
        print(
            f"  Episodes per sampling: {config.training.episodes_per_sampling}")

        # Environment config
        print("\nEnvironment Configuration:")
        print(f"  Training parallel envs: {config.vector_env.num_envs}")
        print(
            f"  Async mode: {'Enabled' if config.vector_env.async_env else 'Disabled'}")

        # Evaluation config
        print("\nEvaluation Configuration:")
        print(
            f"  Training-time evaluation: {'Enabled' if config.training.enable_training_eval else 'Disabled'}")
        if config.training.enable_training_eval:
            print(f"  Eval frequency: every {config.training.eval_freq} steps")
            print(f"  Test episodes: {config.training.num_eval_episodes}")
            print(f"  Test envs: {config.training.num_eval_envs}")
            print(
                f"  Eval policy: {'Pure greedy (epsilon=0)' if config.training.eval_epsilon == 0.0 else f'epsilon-greedy (epsilon={config.training.eval_epsilon})'}")

        # System config
        print("\nSystem Configuration:")
        print(f"  Training device: {config.device}")
        print(f"  Model save dir: {config.save_dir}")
        print(f"  Log dir: {config.log_dir}")

        # Frequency settings
        print("\nLogging Frequency:")
        print(f"  Eval frequency: every {config.training.eval_freq} steps")
        print(f"  Save frequency: every {config.training.save_freq} steps")
        print(
            f"  Graph pool update: every {config.training.graph_pool_update_freq} steps")

        print("=" * 60)

    def _create_vectorized_environment(self):
        """Create batch environment for training."""
        self.logger.info("Creating training vectorized environment via gym...")
        variant = self.config.variant.value
        num_envs = self.config.vector_env.num_envs

        # Get full environment parameters from config
        env_kwargs = {
            'max_nodes': self.config.network.max_nodes,
            'min_nodes': self.config.network.min_nodes,
            'aux_dim': self.config.model.aux_dim,
            'graph_type': self.config.network.graph_type,
            'training_type': getattr(self.config.network, 'training_type', 'uniform'),
            'use_graph_pool': True,  # Always use graph pool
        }

        self.logger.info(f"Training environment config: {env_kwargs}")

        # Batch environment adapter based on gym.vector
        vec_env = make_gym_batch_env(
            variant,
            batch_size=num_envs,
            async_env=self.config.vector_env.async_env,
            **env_kwargs
        )
        return vec_env

    def _create_test_environment(self, num_test_envs: int = None):
        """Create vectorized environment dedicated for test metric recording (independent of training env)."""
        if num_test_envs is None:
            num_test_envs = self.config.training.num_eval_envs

        self.logger.info(
            f"Creating test vectorized environment with {num_test_envs} envs...")
        self.logger.info(
            f"Target evaluation episodes: {self.config.training.num_eval_episodes}")
        variant = self.config.variant.value

        # Test environment uses the same config but is created independently
        env_kwargs = {
            'max_nodes': self.config.network.max_nodes,
            'min_nodes': self.config.network.min_nodes,
            'aux_dim': self.config.model.aux_dim,
            'graph_type': self.config.network.graph_type,
            'training_type': getattr(self.config.network, 'training_type', 'uniform'),
            'use_graph_pool': True,  # Use independent graph pool
        }

        self.logger.info(f"Test environment config: {env_kwargs}")

        # Create independent test environment
        test_vec_env = make_gym_batch_env(
            variant,
            batch_size=num_test_envs,
            async_env=self.config.vector_env.async_env,
            **env_kwargs
        )
        return test_vec_env

    def _create_replay_buffer(self):
        """Create experience replay buffer."""
        self.logger.info("Creating replay buffer...")

        buffer_type = 'prioritized' if self.config.training.is_prioritized_sampling else 'standard'

        if buffer_type == 'prioritized':
            buffer = create_replay_buffer(
                buffer_type='prioritized',
                capacity=self.config.training.memory_size,
                n_step=self.config.training.n_step,
                gamma=self.config.training.gamma,
                alpha=self.config.training.priority_alpha,
                beta=self.config.training.priority_beta,
                beta_increment=self.config.training.priority_beta_increment,
                epsilon=self.config.training.priority_epsilon,
                td_err_upper=getattr(self.config.training,
                                     'td_err_upper', None),
                device=str(self.device)
            )
        else:
            buffer = create_replay_buffer(
                buffer_type='standard',
                capacity=self.config.training.memory_size,
                n_step=self.config.training.n_step,
                gamma=self.config.training.gamma,
                device=str(self.device)
            )

        return buffer

    def _create_model_interface(self):
        """Create and initialize policy network and optimizer."""
        # Map trainers.config.ModelConfig to models.gnn_arch.FinderConfig
        mcfg = self.config.model
        fcfg = FinderConfig()
        fcfg.EMBEDDING_SIZE = mcfg.embedding_size
        fcfg.REG_HIDDEN = mcfg.reg_hidden
        fcfg.AUX_DIM = mcfg.aux_dim
        fcfg.MAX_BP_ITER = mcfg.max_bp_iter
        fcfg.AGGREGATOR_ID = mcfg.aggregator_id
        fcfg.EMBEDDING_METHOD = mcfg.embedding_method
        fcfg.INITIALIZATION_STDDEV = mcfg.initialization_stddev
        fcfg.ALPHA = self.config.training.alpha
        fcfg.USE_HUBER = self.config.training.is_huber_loss

        # Determine DQN type based on configuration
        if self.config.training.is_dueling_dqn:
            # DuelingDQNPolicyNetwork now inherits from DoubleDQNPolicyNetwork
            # so it automatically benefits from Double DQN
            dqn_type = 'dueling'
        elif self.config.training.is_double_dqn:
            dqn_type = 'double'
        else:
            dqn_type = 'vanilla'  # Default choice aligned with original FINDER

        self.q_network = create_policy_network(
            variant=self.config.variant.value,
            config=fcfg,
            device=self.device,
            dqn_type=dqn_type
        )
        self.target_network = None

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.training.learning_rate
        )

    def _compute_exploration_rate(self, step: int) -> float:
        """Compute epsilon-greedy exploration rate."""
        eps_start = self.config.training.eps_start
        eps_end = self.config.training.eps_end
        eps_decay = self.config.training.eps_decay_steps

        if step < eps_decay:
            return eps_end + (eps_start - eps_end) * (eps_decay - step) / eps_decay
        else:
            return eps_end

    def _select_actions(self, observations: List[Dict[str, Any]], exploration_rate: float) -> List[int]:
        """Use the policy network for epsilon-greedy action selection, returning node IDs required by the environment."""
        if self.q_network is None:
            self._create_model_interface()

        # Convert environment observations to GraphBatch and construct model input
        batch = create_graph_batch_from_observations(observations)
        batch_dict = {
            'node_features': batch.node_features.to(self.device),
            'edge_index': batch.edge_index.to(self.device),
            'batch': batch.batch.to(self.device),
            'aux_features': batch.aux_features.to(self.device),
        }
        if batch.node_ids is not None:
            batch_dict['node_ids'] = batch.node_ids.to(self.device)

        with torch.no_grad():
            actions = self.q_network.select_action(
                batch_dict, epsilon=exploration_rate, use_target=False)
        return actions

    def _collect_complete_episodes(self, num_episodes: int):
        """Collect complete episodes following original FINDER logic (sampled every 10 training iterations)."""
        # On first call, initialize environment and accumulators
        if not hasattr(self, 'current_observations'):
            self.current_observations, infos = self.vec_env.reset()
            self.episode_rewards_accumulator = [
                0.0] * len(self.current_observations)

        completed_episodes = 0
        observations = self.current_observations

        # Continue sampling until the specified number of episodes is complete
        while completed_episodes < num_episodes:
            # Select actions (node IDs)
            exploration_rate = self._compute_exploration_rate(
                self.current_iteration)
            actions = self._select_actions(observations, exploration_rate)

            # Execute environment step
            next_observations, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions)

            # Handle automatic reset mechanism of vectorized environments
            for env_id in range(len(observations)):
                obs = observations[env_id]
                done_flag = bool(terminateds[env_id] or truncateds[env_id])

                # Accumulate reward
                self.episode_rewards_accumulator[env_id] += float(
                    rewards[env_id])

                if done_flag:
                    # Environment has ended and auto-reset; get the true terminal state from infos
                    completed_episodes += 1  # Complete episode count +1

                    if ('final_observation' in infos and
                        len(infos['final_observation']) > env_id and
                            infos['final_observation'][env_id] is not None):
                        final_obs = infos['final_observation'][env_id]
                        final_info = infos.get(
                            'final_info', [{}] * len(observations))[env_id]

                        # Add terminal transition to replay buffer
                        self.replay_buffer.add_experience(
                            env_id=env_id,
                            state=obs,
                            action=actions[env_id],
                            reward=float(rewards[env_id]),
                            next_state=final_obs,
                            done=True,
                            graph_data={'obs': obs, 'next_obs': final_obs},
                            aux_features=obs.get(
                                'aux_features', np.zeros(4, dtype=np.float32))
                        )
                    else:
                        # Fallback: if no final_observation, use current observation
                        self.replay_buffer.add_experience(
                            env_id=env_id,
                            state=obs,
                            action=actions[env_id],
                            reward=float(rewards[env_id]),
                            next_state=next_observations[env_id],
                            done=True,
                            graph_data={'obs': obs,
                                        'next_obs': next_observations[env_id]},
                            aux_features=obs.get(
                                'aux_features', np.zeros(4, dtype=np.float32))
                        )

                    # Record episode reward and reset accumulator
                    episode_reward = self.episode_rewards_accumulator[env_id]
                    self.training_stats['episode_rewards'].append(
                        episode_reward)
                    self.episode_rewards_accumulator[env_id] = 0.0
                    self.episode_count += 1

                else:
                    # Add intermediate transition to replay buffer
                    self.replay_buffer.add_experience(
                        env_id=env_id,
                        state=obs,
                        action=actions[env_id],
                        reward=float(rewards[env_id]),
                        next_state=next_observations[env_id],
                        done=False,
                        graph_data={'obs': obs,
                                    'next_obs': next_observations[env_id]},
                        aux_features=obs.get('aux_features', np.zeros(4))
                    )

            # Update observation state and accumulated environment steps
            observations = next_observations
            self.global_step += len(observations)

        # Save current observation state for next call
        self.current_observations = observations

    def _collect_experiences(self, num_steps: int):
        """Collect experiences from batch environment while handling vector env auto-reset.
        Note: This method is now superseded by _collect_complete_episodes, kept for compatibility."""
        # On first call, initialize environment and accumulators
        if not hasattr(self, 'current_observations'):
            self.current_observations, infos = self.vec_env.reset()
            self.episode_rewards_accumulator = [
                0.0] * len(self.current_observations)

        observations = self.current_observations

        for _ in range(num_steps):
            # Select actions (node IDs)
            # Note: use current training iteration (not accumulated env steps) to compute epsilon
            exploration_rate = self._compute_exploration_rate(
                self.current_iteration)
            actions = self._select_actions(observations, exploration_rate)

            # Execute environment step
            next_observations, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions)

            # Handle automatic reset mechanism of vectorized environments
            for env_id in range(len(observations)):
                obs = observations[env_id]
                done_flag = bool(terminateds[env_id] or truncateds[env_id])

                # Accumulate reward
                self.episode_rewards_accumulator[env_id] += float(
                    rewards[env_id])

                if done_flag:
                    # Environment has ended and auto-reset; get the true terminal state from infos
                    # next_observations[env_id] is already the initial observation after reset
                    if ('final_observation' in infos and
                        infos['final_observation'] is not None and
                            env_id < len(infos['final_observation'])):
                        # Use terminal observation as next state
                        terminal_next_obs = infos['final_observation'][env_id]
                    else:
                        # Fallback: use current observation as terminal state
                        # This can happen in certain gymnasium versions or configurations
                        terminal_next_obs = obs
                        self.logger.debug(
                            f"No final_observation for env {env_id}, using current obs as terminal state")

                    # Add terminal experience to replay buffer
                    aux = obs.get('aux_features', np.zeros(
                        4, dtype=np.float32))
                    self.replay_buffer.add_experience(
                        env_id=env_id,
                        state=obs,
                        action=int(actions[env_id]),
                        reward=float(rewards[env_id]),
                        next_state=terminal_next_obs,
                        done=True,  # Ensure marked as terminated
                        graph_data={'variant': self.config.variant.value},
                        aux_features=aux
                    )

                    # Record complete episode reward
                    episode_reward = self.episode_rewards_accumulator[env_id]
                    self.training_stats['episode_rewards'].append(
                        episode_reward)
                    # Only log once every 10 episodes to avoid excessive logging
                    if self.episode_count % 10 == 0:
                        self.logger.info(
                            f"Episode {self.episode_count} completed, total reward: {episode_reward:.4f}")
                    # Reset accumulator
                    self.episode_rewards_accumulator[env_id] = 0.0
                    self.episode_count += 1
                else:
                    # Environment not done, add experience normally
                    aux = obs.get('aux_features', np.zeros(
                        4, dtype=np.float32))
                    self.replay_buffer.add_experience(
                        env_id=env_id,
                        state=obs,
                        action=int(actions[env_id]),
                        reward=float(rewards[env_id]),
                        next_state=next_observations[env_id],
                        done=False,
                        graph_data={'variant': self.config.variant.value},
                        aux_features=aux
                    )

            # Update observations (next_observations already accounts for auto-reset)
            observations = next_observations
            self.global_step += len(observations)

        # Save current observation state for next call
        self.current_observations = observations

    def _train_step(self) -> float:
        """Execute one training step."""
        if not self.replay_buffer.is_ready(self.config.training.batch_size):
            return 0.0

        # Sample batch data from replay buffer
        sample = self.replay_buffer.sample(self.config.training.batch_size)

        # Convert sampled observations to GraphBatch
        state_batch = create_graph_batch_from_observations(sample.states)
        next_state_batch = create_graph_batch_from_observations(
            sample.next_states)

        # Prepare batch dict
        bd = {
            'node_features': state_batch.node_features.to(self.device),
            'edge_index': state_batch.edge_index.to(self.device),
            'batch': state_batch.batch.to(self.device),
            'aux_features': state_batch.aux_features.to(self.device),
            'laplacian': state_batch.laplacian.to(self.device),
            'edge_weight_sum': state_batch.edge_weight_sum.to(self.device),
        }
        if state_batch.node_ids is not None:
            bd['node_ids'] = state_batch.node_ids.to(self.device)

        next_bd = {
            'node_features': next_state_batch.node_features.to(self.device),
            'edge_index': next_state_batch.edge_index.to(self.device),
            'batch': next_state_batch.batch.to(self.device),
            'aux_features': next_state_batch.aux_features.to(self.device),
        }

        if self.q_network is None:
            self._create_model_interface()

        # Convert actions (node IDs) to per-graph local indices
        actions_global_ids = sample.actions
        local_actions: List[int] = []
        node_offset = 0
        batch_tensor = bd['batch']
        total_nodes = batch_tensor.size(0)
        node_ids_tensor = bd.get('node_ids')
        for g in range(int(batch_tensor.max().item()) + 1 if total_nodes > 0 else 0):
            mask = (batch_tensor == g)
            graph_nodes = int(mask.sum().item())
            if graph_nodes == 0:
                local_actions.append(0)
            else:
                if node_ids_tensor is not None:
                    ids_slice = node_ids_tensor[mask]
                    target_id = int(actions_global_ids[g])
                    # Find local index of target ID within the graph
                    eq = (ids_slice == target_id).nonzero(as_tuple=False)
                    la = int(eq[0].item()) if eq.numel() > 0 else 0
                    local_actions.append(la)
                else:
                    # Fall back to truncated range when no node_ids
                    la = int(
                        np.clip(actions_global_ids[g], 0, max(graph_nodes - 1, 0)))
                    local_actions.append(la)
            node_offset += graph_nodes

        actions_tensor = torch.tensor(
            local_actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            sample.rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(sample.dones, dtype=torch.float32,
                             device=self.device).view(-1, 1)

        # Current Q(pred) (based on selected local actions)
        out_cur = self.q_network.forward(
            {**bd, 'act_idxs': actions_tensor}, use_target=False, return_embeddings=True)
        q_pred = out_cur['q_pred']  # [B, 1]

        # Target Q computed polymorphically by policy network: Vanilla uses target network, Double/Dueling use main net for action selection + target net for valuation
        gamma_bootstrap = float(
            self.config.training.gamma) ** max(1, int(self.config.training.n_step))
        target_q = self.q_network.compute_target_values(
            next_batch_data=next_bd,
            rewards=rewards,
            dones=dones,
            gamma=gamma_bootstrap
        )

        is_weights = None
        if getattr(sample, 'importance_weights', None) is not None:
            is_weights = torch.tensor(
                sample.importance_weights,
                dtype=torch.float32,
                device=self.device
            ).view(-1, 1)

        missing_loss_fields = [key for key in (
            'laplacian', 'edge_weight_sum') if key not in bd]
        if missing_loss_fields:
            raise ValueError(
                f"Graph batch missing loss fields: {missing_loss_fields}")

        # Combined loss (includes reconstruction term)
        loss_dict = self.q_network.loss_fn(
            q_pred=q_pred,
            targets=target_q,
            node_embeddings=out_cur['node_embeddings'],
            laplacian=bd['laplacian'],
            edge_weight_sum=bd['edge_weight_sum'],
            is_weights=is_weights
        )
        loss = loss_dict['total_loss']

        # Prioritized replay: update priorities
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and sample.indices is not None:
            td_errors = torch.abs(
                target_q - q_pred).detach().cpu().numpy().reshape(-1)
            self.replay_buffer.update_priorities(sample.indices, td_errors)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        try:
            max_grad_norm = float(os.environ.get(
                "FINDER_MAX_GRAD_NORM",
                getattr(self.config.training, "max_grad_norm", 5.0)
            ))
        except Exception:
            max_grad_norm = 5.0
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=max_grad_norm)
        self.optimizer.step()

        return float(loss.item())

    def _update_target_network(self):
        """Update the internal target network of the policy."""
        if self.q_network is not None:
            self.q_network.update_target_network()

    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model performance (using dedicated test environment and pure greedy policy, matching original FINDER.Test())."""
        if not self.config.training.enable_training_eval:
            # If training-time evaluation is not enabled, return an approximation from training rewards
            if not self.training_stats['episode_rewards']:
                return {
                    'avg_reward': 0.0,
                    'episodes_evaluated': 0,
                    'eval_method': 'training_rewards'
                }
            recent_rewards = self.training_stats['episode_rewards'][-100:]
            return {
                'avg_reward': float(np.mean(recent_rewards)),
                'episodes_evaluated': len(recent_rewards),
                'eval_method': 'training_rewards'
            }

        if self.test_vec_env is None or self.q_network is None:
            return {'avg_reward': 0.0, 'eval_method': 'fallback'}

        # Evaluate using the dedicated test environment (matching original FINDER's Test function behavior)
        target_episodes = self.config.training.num_eval_episodes
        self.logger.info(
            f"Starting model evaluation (using test environment, pure greedy policy, target episodes: {target_episodes})...")

        self.q_network.eval()
        test_rewards = []
        test_lengths = []

        # Create evaluation progress bar
        eval_pbar = tqdm(total=target_episodes, desc="Eval Progress", leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]')

        try:
            completed_episodes = 0

            # Calculate how many rounds of parallel envs to run to reach target episodes
            if hasattr(self.test_vec_env, 'envs'):
                num_parallel_envs = len(self.test_vec_env.envs)
            elif hasattr(self.test_vec_env, 'num_envs'):
                num_parallel_envs = self.test_vec_env.num_envs
            else:
                num_parallel_envs = self.config.training.num_eval_envs

            while completed_episodes < target_episodes:
                # Calculate number of envs to run this round (avoid exceeding target)
                remaining_episodes = target_episodes - completed_episodes
                current_batch_size = min(num_parallel_envs, remaining_episodes)

                # Reset test environment
                obs_list, _ = self.test_vec_env.reset()
                # Only use the needed number of environments
                obs_list = obs_list[:current_batch_size]

                # Convert to PyG batch data
                batch_obj = create_graph_batch_from_observations(obs_list)
                batch_data = {
                    'node_features': batch_obj.node_features.to(self.device),
                    'edge_index': batch_obj.edge_index.to(self.device),
                    'batch': batch_obj.batch.to(self.device),
                    'aux_features': batch_obj.aux_features.to(self.device),
                    'laplacian': batch_obj.laplacian.to(self.device),
                    'edge_weight_sum': batch_obj.edge_weight_sum.to(self.device),
                    'node_ids': batch_obj.node_ids if hasattr(batch_obj, 'node_ids') else None
                }

                episode_rewards = [0.0] * current_batch_size
                episode_lengths = [0] * current_batch_size
                episodes_done = 0
                max_steps = 1000  # Prevent infinite loop

                for step in range(max_steps):
                    # Use pure greedy policy to select actions (epsilon=0, matching original FINDER.Test())
                    with torch.no_grad():
                        actions = self.q_network.select_action(
                            batch_data,
                            epsilon=self.config.training.eval_epsilon,  # should be 0.0
                            use_target=False
                        )

                    # Only execute actions for the current batch size of environments
                    if len(actions) > current_batch_size:
                        actions = actions[:current_batch_size]

                    # Execute actions
                    obs_list, rewards, terminated, truncated, infos = self.test_vec_env.step(
                        actions)
                    obs_list = obs_list[:current_batch_size]
                    rewards = rewards[:current_batch_size]
                    terminated = terminated[:current_batch_size]
                    truncated = truncated[:current_batch_size]

                    # Update rewards and step lengths
                    for i in range(current_batch_size):
                        episode_rewards[i] += rewards[i]
                        episode_lengths[i] += 1

                        # Check if episode ended
                        if terminated[i] or truncated[i]:
                            test_rewards.append(episode_rewards[i])
                            test_lengths.append(episode_lengths[i])
                            episodes_done += 1

                    # If all episodes are done, break out of inner loop
                    if episodes_done >= current_batch_size:
                        break

                    # Prepare batch data for next step
                    if episodes_done < current_batch_size:
                        batch_obj = create_graph_batch_from_observations(
                            obs_list)
                        batch_data = {
                            'node_features': batch_obj.node_features.to(self.device),
                            'edge_index': batch_obj.edge_index.to(self.device),
                            'batch': batch_obj.batch.to(self.device),
                            'aux_features': batch_obj.aux_features.to(self.device),
                            'laplacian': batch_obj.laplacian.to(self.device),
                            'edge_weight_sum': batch_obj.edge_weight_sum.to(self.device),
                            'node_ids': batch_obj.node_ids if hasattr(batch_obj, 'node_ids') else None
                        }

                # Update completed episode count
                completed_episodes += episodes_done
                eval_pbar.update(episodes_done)

            # Compute evaluation metrics
            if test_rewards:
                avg_reward = float(np.mean(test_rewards))
                avg_length = float(np.mean(test_lengths))
                std_reward = float(np.std(test_rewards))

                self.logger.info(f"Evaluation complete: avg reward={avg_reward:.6f} +/- {std_reward:.6f}, "
                                 f"avg steps={avg_length:.1f}, eval episodes={len(test_rewards)}")

                return {
                    'avg_reward': avg_reward,
                    'std_reward': std_reward,
                    'avg_length': avg_length,
                    'episodes_evaluated': len(test_rewards),
                    'eval_method': 'test_environment_greedy'
                }
            else:
                self.logger.warning("Evaluation did not complete any episodes")
                return {'avg_reward': 0.0, 'eval_method': 'failed'}

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {'avg_reward': 0.0, 'eval_method': 'error'}

        finally:
            eval_pbar.close()
            self.q_network.train()  # Restore training mode

    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'current_iteration': self.current_iteration,
            'model_state_dict': self.q_network.state_dict() if self.q_network else None,
            'target_state_dict': self._get_target_state_dict() if self.q_network else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'best_performance': self.best_performance
        }

        # Ensure save directory exists
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            self.config.save_dir,
            f'checkpoint_{self.config.variant.value}_iter_{iteration}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _get_target_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return internal target network state using prefixed format for compatibility with multi-target submodules."""
        if self.q_network is None:
            return {}

        state = {
            f'target_gnn.{key}': value
            for key, value in self.q_network.target_gnn.state_dict().items()
        }
        if hasattr(self.q_network, 'target_value_stream'):
            state.update({
                f'target_value_stream.{key}': value
                for key, value in self.q_network.target_value_stream.state_dict().items()
            })
        return state

    def _load_model_state_dict(self, model_state_dict: Dict[str, torch.Tensor]):
        """Load model state; compatible with old Dueling checkpoints missing target_value_stream."""
        if self.q_network is None or not model_state_dict:
            return

        try:
            self.q_network.load_state_dict(model_state_dict)
            return
        except RuntimeError:
            incompatible = self.q_network.load_state_dict(
                model_state_dict, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            allowed_missing = [
                key for key in missing if key.startswith('target_value_stream.')]
            bad_missing = [
                key for key in missing if key not in allowed_missing]
            if bad_missing or unexpected:
                raise
            if hasattr(self.q_network, 'target_value_stream'):
                self.q_network.target_value_stream.load_state_dict(
                    self.q_network.value_stream.state_dict())

    def _load_target_state_dict(self, target_state_dict: Optional[Dict[str, torch.Tensor]]):
        """Load internal target network state, compatible with new format, bare target_gnn format, and old external policy format."""
        if self.q_network is None:
            return
        if not target_state_dict:
            self.q_network.update_target_network()
            return

        def strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            return {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

        # Old checkpoint target_state_dict comes from external target_network.state_dict(),
        # where gnn.* represents the target valuation network saved at that time; new format only saves target_gnn.*.
        target_gnn_state = strip_prefix(target_state_dict, 'gnn.')
        if not target_gnn_state:
            target_gnn_state = strip_prefix(target_state_dict, 'target_gnn.')
        if not target_gnn_state:
            target_gnn_state = target_state_dict
        self.q_network.target_gnn.load_state_dict(target_gnn_state)

        if hasattr(self.q_network, 'target_value_stream'):
            target_value_state = strip_prefix(
                target_state_dict, 'target_value_stream.')
            if not target_value_state:
                target_value_state = strip_prefix(
                    target_state_dict, 'value_stream.')
            if target_value_state:
                self.q_network.target_value_stream.load_state_dict(
                    target_value_state)
            else:
                self.q_network.target_value_stream.load_state_dict(
                    self.q_network.value_stream.state_dict())

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.current_iteration = checkpoint.get(
            'current_iteration', 0)  # Backward compatibility
        self.training_stats = checkpoint['training_stats']
        self.best_performance = checkpoint['best_performance']

        if self.q_network is None:
            self._create_model_interface()

        self._load_model_state_dict(checkpoint.get('model_state_dict'))
        self._load_target_state_dict(checkpoint.get('target_state_dict'))
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint['iteration']

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            resume_from: Checkpoint path to resume from

        Returns:
            Training statistics and final performance metrics
        """
        self.logger.info("Starting FINDER training...")

        # Resume from checkpoint if provided
        start_iteration = 0
        if resume_from:
            start_iteration = self._load_checkpoint(resume_from)

        # Initialize model if not loaded from checkpoint
        if self.q_network is None:
            self._create_model_interface()

        # Initialize progress bar statistics variables
        recent_rewards = []
        recent_losses = []
        recent_episodes = 0

        # Training readiness flag
        training_ready = False

        # Training loop
        with tqdm(range(start_iteration, self.config.training.max_iterations),
                  desc=f"Training {self.config.variant.value.upper()}",
                  unit="iter") as pbar:
            for iteration in pbar:
                # Update current iteration (used for epsilon decay)
                self.current_iteration = iteration

                # Follow original FINDER logic: sample at configured frequency
                if iteration % self.config.training.sampling_freq == 0:
                    # Collect the configured number of complete episodes
                    episodes_before = self.episode_count
                    self._collect_complete_episodes(
                        self.config.training.episodes_per_sampling)
                    episodes_after = self.episode_count
                    new_episodes = episodes_after - episodes_before
                    recent_episodes += new_episodes
                else:
                    new_episodes = 0

                # Check training readiness (only on first check)
                if not training_ready:
                    training_ready = self.replay_buffer.is_ready(
                        self.config.training.batch_size)
                    if training_ready:
                        self.logger.info(
                            f"Replay buffer ready, starting formal training (iteration {iteration})")

                # Training step
                loss = 0.0
                if training_ready:
                    loss = self._train_step()
                    self.training_stats['loss_history'].append(loss)
                    recent_losses.append(loss)

                    # Record training metrics
                    if iteration % 100 == 0:
                        exploration_rate = self._compute_exploration_rate(
                            iteration)
                        self.training_stats['exploration_rate'].append(
                            exploration_rate)
                        self.training_stats['replay_buffer_size'].append(
                            len(self.replay_buffer))

                        self.writer.add_scalar(
                            'Training/Loss', loss, iteration)
                        self.writer.add_scalar(
                            'Training/ExplorationRate', exploration_rate, iteration)
                        self.writer.add_scalar(
                            'Training/ReplayBufferSize', len(self.replay_buffer), iteration)
                        self.writer.add_scalar(
                            'Training/EpisodeCount', self.episode_count, iteration)

                # Update progress bar statistics every 10 steps
                # Note: sampling now also occurs every 10 steps, so info update is synchronized with sampling
                if iteration % 10 == 0:
                    # Compute average metrics - show when loss data exists, otherwise show None
                    if recent_losses:
                        avg_loss = float(np.mean(recent_losses[-100:]))
                        loss_display = f'{avg_loss:.4f}'
                    else:
                        loss_display = 'None'

                    # Compute reward statistics - show when completed episodes exist, otherwise show None
                    num_rewards = len(self.training_stats['episode_rewards'])
                    if num_rewards > 0:
                        recent_rewards = self.training_stats['episode_rewards'][-50:]
                        avg_reward = float(np.mean(recent_rewards))
                        reward_display = f'{avg_reward:.3f}'
                    else:
                        reward_display = 'None'

                    exploration_rate = self._compute_exploration_rate(
                        iteration)
                    buffer_size = len(self.replay_buffer)

                    # Compute total training episode progress
                    total_sampling_times = self.config.training.max_iterations // self.config.training.sampling_freq + 1
                    total_expected_episodes = total_sampling_times * \
                        self.config.training.episodes_per_sampling
                    sampling_progress = f"{self.episode_count}/{total_expected_episodes}"

                    # Update progress bar description
                    pbar.set_postfix({
                        'Loss': loss_display,
                        'Reward': reward_display,
                        'Eps': f'{exploration_rate:.3f}',
                        'Episodes': sampling_progress,
                        'BufferSize': f'{buffer_size:,}',
                        'Steps': f'{self.global_step:,}'
                    }, refresh=False)

                    # Reset temporary statistics variables
                    if len(recent_losses) > 200:  # Keep the most recent 200 loss values
                        recent_losses = recent_losses[-100:]

                # Update target network (only after training is ready)
                if (training_ready and
                    iteration > 0 and
                        iteration % self.config.training.update_target_freq == 0):
                    self._update_target_network()
                    self.logger.info(
                        f"Target network updated at iteration {iteration}")

                # Evaluation (only after training is ready)
                if (training_ready and
                    iteration > 0 and
                        iteration % self.config.training.eval_freq == 0):
                    eval_metrics = self._evaluate_model()
                    current_performance = eval_metrics.get('avg_reward', 0.0)

                    self.logger.info(f"Iteration {iteration}: {eval_metrics}")

                    # Save best model
                    if current_performance > self.best_performance:
                        self.best_performance = current_performance
                        os.makedirs(self.config.save_dir, exist_ok=True)
                        best_model_path = os.path.join(
                            self.config.save_dir,
                            f'best_model_{self.config.variant.value}.pt'
                        )
                        if self.q_network:
                            torch.save(self.q_network.state_dict(),
                                       best_model_path)
                        self.logger.info(
                            f"New best performance: {current_performance:.6f}")

                    # Log evaluation metrics (only numeric types)
                    for key, value in eval_metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(
                                f'Evaluation/{key}', value, iteration)

                # Save checkpoint (only after training is ready)
                if (training_ready and
                    iteration > 0 and
                        iteration % self.config.training.save_freq == 0):
                    self._save_checkpoint(iteration)

                # Periodically update graph pool (at configured update frequency)
                if iteration > 0 and iteration % self.config.training.graph_pool_update_freq == 0:
                    self.logger.info(
                        f"Iteration {iteration}: Updating graph pool...")
                    try:
                        regenerate_all_pools()
                        # Record graph pool statistics
                        pool_stats = get_all_pool_stats()
                        for pool_name, stats in pool_stats.items():
                            self.logger.info(f"Graph pool {pool_name}: size={stats['pool_size']}, "
                                             f"gen_count={stats['generation_count']}, "
                                             f"sample_count={stats['sample_count']}")
                            self.writer.add_scalar(
                                f'GraphPool/{pool_name}_samples', stats['sample_count'], iteration)
                        self.logger.info("Graph pool update complete")
                    except Exception as e:
                        self.logger.warning(f"Graph pool update failed: {e}")
                        # Continue training without interruption

        # Final evaluation and cleanup
        final_metrics = self._evaluate_model()
        self.logger.info(f"Training complete. Final metrics: {final_metrics}")

        # Save final model
        os.makedirs(self.config.save_dir, exist_ok=True)
        final_model_path = os.path.join(
            self.config.save_dir,
            f'final_model_{self.config.variant.value}.pt'
        )
        if self.q_network:
            torch.save(self.q_network.state_dict(), final_model_path)

        # Close environments and writer
        self.vec_env.close()
        if self.test_vec_env is not None:
            self.test_vec_env.close()
        self.writer.close()

        return {
            'final_performance': self.best_performance,
            'total_episodes': self.episode_count,
            'total_steps': self.global_step,
            'training_stats': self.training_stats
        }


# Utility functions for creating and running training

def create_trainer(
    variant: FinderVariant,
    config_overrides: Optional[Dict[str, Any]] = None,
    env_class: Optional[type] = None,
    model_class: Optional[type] = None
) -> FinderVectorTrainer:
    """
    Factory function to create a FINDER trainer.

    Args:
        variant: FINDER variant to train
        config_overrides: Configuration overrides
        env_class: Optional custom environment class
        model_class: Optional custom model class

    Returns:
        Configured trainer instance
    """
    from .config import get_default_config

    config = get_default_config(variant)

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.network, key):
                setattr(config.network, key, value)
            elif hasattr(config.vector_env, key):
                setattr(config.vector_env, key, value)

    return FinderVectorTrainer(config, env_class, model_class)


def run_training_experiment(
    variant: FinderVariant,
    experiment_name: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    env_class: Optional[type] = None,
    model_class: Optional[type] = None,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a complete training experiment.

    Args:
        variant: FINDER variant
        experiment_name: Experiment name
        config_overrides: Configuration overrides
        env_class: Optional custom environment class
        model_class: Optional custom model class
        resume_from: Checkpoint to resume from

    Returns:
        Training results
    """
    trainer = create_trainer(variant, config_overrides, env_class, model_class)

    # Save config
    config_path = os.path.join(trainer.config.save_dir, 'config.json')
    trainer.config.save(config_path)

    # Run training
    results = trainer.train(resume_from)

    # Save results
    results_path = os.path.join(trainer.config.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results
