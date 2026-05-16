#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Experience Replay Buffer System

This module provides experience replay functionality adapted from the original
FINDER nstep_replay_mem.pyx implementation, but using pure Python and modern deep
learning libraries.

Replay buffers support:
- N-step learning for improved sampling efficiency
- Prioritized experience replay (optional)
- Graph-based state representations
- Batch sampling for training
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
import torch
from dataclasses import dataclass
from .sum_tree import SumTree, MinTree


@dataclass
class Experience:
    """Single experience transition"""
    state: Any  # Graph state representation
    action: int  # Executed action
    reward: float  # Immediate reward
    next_state: Any  # Next graph state
    done: bool  # Terminal flag
    graph_data: Dict[str, Any]  # Additional graph information
    aux_features: np.ndarray  # Auxiliary features (node degrees, etc.)


class ReplaySample(NamedTuple):
    """Batch of samples drawn from the replay buffer"""
    states: List[Any]
    actions: List[int]
    rewards: List[float]
    next_states: List[Any]
    dones: List[bool]
    graph_data_list: List[Dict[str, Any]]
    aux_features_list: List[np.ndarray]
    indices: Optional[List[int]] = None  # For prioritized replay
    importance_weights: Optional[np.ndarray] = None  # For prioritized replay


class NStepReplayBuffer:
    """
    N-step experience replay buffer

    Implements n-step learning, where returns are computed over n time steps
    rather than just immediate rewards. This can improve sampling efficiency.

    Based on the original FINDER nstep_replay_mem.pyx implementation.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 5,
        gamma: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            n_step: Number of steps for n-step learning
            gamma: Discount factor for future rewards
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.device = device

        # Storage (ring buffer)
        self.storage = [None] * capacity  # Fixed-capacity ring buffer
        # Logical index for next write (cycles 0..capacity-1)
        self.next_index = 0
        self.storage_size = 0             # Actual number written (<= capacity)
        # Logical index of the most recent write
        self.last_added_index: Optional[int] = None
        self.n_step_buffers = {}  # Per-environment n-step buffers

        # Statistics
        self.total_added = 0

    def add_experience(
        self,
        env_id: int,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        graph_data: Dict[str, Any],
        aux_features: np.ndarray
    ):
        """
        Add a new experience to the buffer.

        Args:
            env_id: Environment identifier for parallel environments
            state: Current state
            action: Executed action
            reward: Immediate reward
            next_state: Resulting state
            done: Whether the episode terminated
            graph_data: Graph-specific data (adjacency lists, etc.)
            aux_features: Auxiliary features of the state
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            graph_data=graph_data,
            aux_features=aux_features
        )

        # Initialize n-step buffer for this environment if needed
        if env_id not in self.n_step_buffers:
            # Use a list to simulate a fixed window (manually maintain length)
            self.n_step_buffers[env_id] = []

        # Add to n-step buffer
        buf = self.n_step_buffers[env_id]
        buf.append(experience)
        # Keep window size no larger than n_step
        if len(buf) > self.n_step:
            del buf[0]

        # If there are enough steps or the episode ended, create n-step experience
        if len(self.n_step_buffers[env_id]) == self.n_step or done:
            n_step_experience = self._create_n_step_experience(
                self.n_step_buffers[env_id])
            self._push_to_storage(n_step_experience)
            self.total_added += 1

            # If episode ended, clear n-step buffer
            if done:
                self.n_step_buffers[env_id].clear()

    def _create_n_step_experience(self, n_step_buffer: list) -> Experience:
        """Create n-step experience from a sequence of experiences."""
        first_exp = n_step_buffer[0]
        last_exp = n_step_buffer[-1]

        # Compute n-step return
        n_step_return = 0.0
        for i, exp in enumerate(n_step_buffer):
            n_step_return += (self.gamma ** i) * exp.reward

        # Create n-step experience
        return Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=n_step_return,
            next_state=last_exp.next_state,
            done=last_exp.done,
            graph_data=first_exp.graph_data,
            aux_features=first_exp.aux_features
        )

    def sample(self, batch_size: int) -> ReplaySample:
        """Sample a batch of experiences."""
        if self.storage_size < batch_size:
            batch_size = self.storage_size

        # Randomly sample from the ring buffer, ensuring consistency with logical indices
        if self.storage_size == 0:
            experiences = []
        else:
            idxs = random.sample(range(self.storage_size), batch_size)
            experiences = [self.storage[i] for i in idxs]

        return ReplaySample(
            states=[exp.state for exp in experiences],
            actions=[exp.action for exp in experiences],
            rewards=[exp.reward for exp in experiences],
            next_states=[exp.next_state for exp in experiences],
            dones=[exp.done for exp in experiences],
            graph_data_list=[exp.graph_data for exp in experiences],
            aux_features_list=[exp.aux_features for exp in experiences]
        )

    def __len__(self) -> int:
        return self.storage_size

    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for training."""
        return self.storage_size >= batch_size

    # Internal: push experience into ring buffer, and record the most recent index
    def _push_to_storage(self, exp: Experience) -> int:
        idx = self.next_index
        self.storage[idx] = exp
        self.last_added_index = idx
        self.next_index = (self.next_index + 1) % self.capacity
        self.storage_size = min(self.storage_size + 1, self.capacity)
        self._on_storage_pushed(idx)
        return idx

    def _on_storage_pushed(self, index: int) -> None:
        """Subclasses can override this hook to sync auxiliary structures after a sample is actually written to the ring buffer."""
        return


class PrioritizedReplayBuffer(NStepReplayBuffer):
    """
    Prioritized experience replay buffer

    Implements priority-based sampling based on TD-error magnitude.
    Samples with higher TD errors are more likely to be selected.

    Based on the original FINDER nstep_replay_mem_prioritized.pyx implementation.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 5,
        gamma: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = 'cuda',
        td_err_upper: Optional[float] = None
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            n_step: N-step learning parameter
            gamma: Discount factor
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
            beta_increment: Increment for beta per sampling
            epsilon: Small constant to avoid zero priority
            device: Computation device
        """
        super().__init__(capacity, n_step, gamma, device)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.td_err_upper = td_err_upper

        # Data structures for prioritized sampling (decoupled from replay logic)
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)
        self.max_raw_priority = 1.0
        self.data_index = 0

    def add_experience(
        self,
        env_id: int,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        graph_data: Dict[str, Any],
        aux_features: np.ndarray
    ):
        """Add experience; new sample priorities are initialized by _on_storage_pushed using the actual write index."""
        super().add_experience(
            env_id, state, action, reward, next_state, done, graph_data, aux_features
        )

    def _update_priority_raw(self, index: int, raw_priority: float):
        """Update tree with raw TD-error priority; internally applies alpha exponent uniformly."""
        raw = max(float(raw_priority), float(self.epsilon))
        if self.td_err_upper is not None:
            raw = min(raw, float(self.td_err_upper))
        p = raw ** self.alpha
        self.sum_tree.update(index, p)
        self.min_tree.update(index, p)
        self.max_raw_priority = max(self.max_raw_priority, raw)

    def _on_storage_pushed(self, index: int) -> None:
        """Each newly written sample is initialized with the current maximum raw priority."""
        self.sum_tree.logical_size = self.storage_size
        self.min_tree.logical_size = self.storage_size
        self._update_priority_raw(index, self.max_raw_priority)
        self.data_index = (index + 1) % self.capacity

    def sample(self, batch_size: int) -> ReplaySample:
        """Sample a batch using priority-based sampling."""
        if self.storage_size == 0:
            return ReplaySample([], [], [], [], [], [], [], indices=[], importance_weights=np.array([], dtype=np.float32))
        if self.storage_size < batch_size:
            batch_size = self.storage_size

        self.sum_tree.logical_size = self.storage_size
        self.min_tree.logical_size = self.storage_size

        # Sample indices based on priority
        total = self.sum_tree.total()
        total = max(1e-12, total)
        segment = total / batch_size

        indices = []
        priorities = []
        for i in range(batch_size):
            min_val = segment * i
            max_val = segment * (i + 1)
            value = random.uniform(min_val, max_val)

            # Sample on SumTree by interval prefix sum
            index = self.sum_tree.get_prefix_sum_index(value)
            index = max(0, min(index, self.storage_size - 1))
            indices.append(index)
            # Actual leaf priority
            priorities.append(max(1e-12, self.sum_tree.get_leaf_value(index)))

        # Compute importance sampling weights
        min_priority = max(1e-12, self.min_tree.minimum())
        N = max(1, self.storage_size)
        max_weight = (min_priority / total * N) ** (-self.beta)

        weights = []
        for priority in priorities:
            weight = (priority / total * N) ** (-self.beta)
            weights.append(weight / max_weight)

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Read directly from ring buffer, ensuring consistency with SumTree indices
        experiences = [self.storage[i] for i in indices]
        assert len(experiences) == len(indices) == len(weights), (
            f"PER sample length mismatch: {len(experiences)} vs {len(indices)} vs {len(weights)}"
        )

        return ReplaySample(
            states=[exp.state for exp in experiences],
            actions=[exp.action for exp in experiences],
            rewards=[exp.reward for exp in experiences],
            next_states=[exp.next_state for exp in experiences],
            dones=[exp.done for exp in experiences],
            graph_data_list=[exp.graph_data for exp in experiences],
            aux_features_list=[exp.aux_features for exp in experiences],
            indices=indices,
            importance_weights=np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.asarray(td_errors).reshape(-1)
        for i, td_error in zip(indices, td_errors):
            raw_priority = abs(float(td_error)) + self.epsilon
            self._update_priority_raw(int(i), raw_priority)


def create_replay_buffer(
    buffer_type: str,
    capacity: int,
    n_step: int = 5,
    gamma: float = 1.0,
    device: str = 'cuda',
    **kwargs
) -> NStepReplayBuffer:
    """
    Factory function to create a replay buffer.

    Args:
        buffer_type: 'standard' or 'prioritized'
        capacity: Buffer capacity
        n_step: N-step learning parameter
        gamma: Discount factor
        device: Computation device
        **kwargs: Additional keyword arguments for the prioritized buffer

    Returns:
        Replay buffer instance
    """
    if buffer_type == 'standard':
        return NStepReplayBuffer(capacity, n_step, gamma, device)
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            capacity, n_step, gamma, device=device, **kwargs
        )
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")
