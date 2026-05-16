#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch environment implemented using gymnasium.vector, without modifying the return values of individual environments (keeping dict observations).

Key points:
- No wrappers, no observation conversion; directly pass through the dict returned by individual environments to the vector environment.
- By passing observation_space=None / action_space=None and setting shared_memory=False,
  let gymnasium pass Python objects (lists/dicts) as-is in non-shared-memory mode.
"""

from functools import partial
from typing import List, Optional, Callable, Union
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import networkx as nx
import numpy as np

from . import ENVIRONMENT_REGISTRY, make_env


class AnySpace(gym.spaces.Space):
    """
    Placeholder space: shape=None, dtype=object; used to satisfy the gym interface without constraining structure.
    """

    def __init__(self):
        super().__init__(shape=None, dtype=object)

    def sample(self):
        return None

    def contains(self, x) -> bool:
        return True

    def __repr__(self) -> str:
        return "AnySpace(shape=None, dtype=object)"

    # Ensure space consistency checks in vector environments pass: any two AnySpace instances are considered equal
    def __eq__(self, other) -> bool:
        return isinstance(other, AnySpace)


class SpacePatchedEnv(gym.Wrapper):
    """Patch arbitrary dict observations/actions with vector-env compatible spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = AnySpace()
        self.observation_space = AnySpace()


def _make_space_patched_env(env_type: str, env_kwargs: dict):
    base_env = make_env(env_type, **env_kwargs)
    return SpacePatchedEnv(base_env)


class GymBatchAdapter:
    """Batch environment adapter based on gymnasium.vector (zero observation conversion)."""

    def __init__(self, env_type: str, batch_size: int, async_env: bool = True, **env_kwargs):
        if env_type not in ENVIRONMENT_REGISTRY:
            raise ValueError(f"Unknown environment type: {env_type}")

        env_fns: List[Callable[[], gym.Env]] = [
            partial(_make_space_patched_env, env_type, dict(env_kwargs))
            for _ in range(batch_size)
        ]

        # Use gym vectorized environment, do not pass observation_space/action_space arguments
        # Disable shared memory in async mode to avoid object stacking issues
        if async_env:
            self.vec = AsyncVectorEnv(env_fns, shared_memory=False)
        else:
            self.vec = SyncVectorEnv(env_fns)

        self.batch_size = batch_size

    def reset(self, graphs: Optional[List[nx.Graph]] = None, seed: Optional[Union[int, List[int]]] = None):
        # Support passing individual options per env (e.g., custom graphs)
        options = [{'graph': g}
                   for g in graphs] if graphs is not None else None
        observations, infos = self.vec.reset(seed=seed, options=options)
        # Normalize obs -> list (keep infos passed through as-is)
        if not isinstance(observations, list):
            observations = list(observations)
        return observations, infos

    def step(self, actions: List[int]):
        observations, rewards, terminated, truncated, infos = self.vec.step(
            actions)
        if not isinstance(observations, list):
            observations = list(observations)
        return observations, rewards, terminated, truncated, infos

    def render(self, indices: Optional[List[int]] = None):
        return None

    def close(self):
        self.vec.close()


def make_gym_batch_env(env_type: str, batch_size: int = 64, async_env: bool = True, **env_kwargs):
    """Factory function: create a gym.vector-based batch environment (zero observation conversion)."""
    return GymBatchAdapter(env_type, batch_size, async_env=async_env, **env_kwargs)
