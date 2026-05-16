#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary data structures for prioritized experience replay: SumTree and MinTree

This module decouples "data structures (trees)" from "sampling/replay logic",
providing standalone SumTree/MinTree implementations for `PrioritizedReplayBuffer`.
"""

from typing import Optional


class _BaseTree:
    """Base class: Array-based storage format using a complete binary tree.

    Leaf range: [leaf_start, leaf_end] = [capacity-1, 2*capacity-2]
    Tree capacity is expanded to the smallest power of 2 no less than capacity.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        # Expand capacity to power of 2 for convenient array-based complete binary tree
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        # Logical capacity (actual number of usable leaves, <= capacity)
        self.logical_size = capacity
        # Array storing tree structure (internal nodes and leaves)
        self.tree = [0.0] * (2 * self.capacity - 1)

    def _leaf_index(self, data_index: int) -> int:
        # Map logical data_index to leaf node index
        if data_index < 0:
            raise ValueError("data_index must be non-negative")
        data_index = data_index % self.logical_size
        return data_index + self.capacity - 1

    def _parent(self, idx: int) -> int:
        return (idx - 1) // 2 if idx > 0 else 0

    def _left(self, idx: int) -> int:
        return 2 * idx + 1

    def _right(self, idx: int) -> int:
        return 2 * idx + 2


class SumTree(_BaseTree):
    """SumTree for proportional-to-weight interval sampling."""

    def total(self) -> float:
        return self.tree[0]

    def update(self, data_index: int, value: float) -> None:
        if value < 0:
            value = 0.0
        idx = self._leaf_index(data_index)
        change = value - self.tree[idx]
        self.tree[idx] = value
        # Propagate updates upward to parent nodes
        while idx != 0:
            idx = self._parent(idx)
            self.tree[idx] += change

    def get_leaf_value(self, data_index: int) -> float:
        """Read the current value of a leaf for a given logical index. Returns 0.0 if unwritten."""
        idx = self._leaf_index(data_index)
        return float(self.tree[idx])

    def get_prefix_sum_index(self, prefix_sum: float) -> int:
        """Given a prefix sum, return the corresponding leaf logical index (data_index)."""
        if prefix_sum < 0:
            prefix_sum = 0.0
        idx = 0
        # Walk down to a leaf
        while True:
            left = self._left(idx)
            right = left + 1
            if left >= len(self.tree):
                break
            # Use strictly less than to ensure boundary values fall into the right interval ([0,left), [left,total))
            if prefix_sum < self.tree[left]:
                idx = left
            else:
                prefix_sum -= self.tree[left]
                idx = right
        # Convert back to logical index
        data_index = idx - (self.capacity - 1)
        # Constrain to logical capacity range
        if data_index < 0:
            data_index = 0
        elif data_index >= self.logical_size:
            data_index = self.logical_size - 1
        return data_index


class MinTree(_BaseTree):
    """MinTree for quickly obtaining the current minimum priority value."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        # Initialize with positive infinity to avoid unwritten positions affecting the minimum
        self.tree = [float("inf")] * (2 * self.capacity - 1)

    def minimum(self) -> float:
        return self.tree[0]

    def update(self, data_index: int, value: float) -> None:
        idx = self._leaf_index(data_index)
        self.tree[idx] = value
        # Propagate updates upward to parent node minima
        while idx != 0:
            idx = self._parent(idx)
            left = self._left(idx)
            right = left + 1
            self.tree[idx] = min(self.tree[left], self.tree[right])
