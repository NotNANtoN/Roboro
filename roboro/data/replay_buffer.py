"""Core replay buffer — uniform sampling, fixed-size ring buffer."""

from __future__ import annotations

import random

import torch

from roboro.core.types import Batch


class ReplayBuffer:
    """Simple uniform experience replay buffer.

    Stores transitions in flat torch tensors.  Once full, oldest transitions
    are overwritten (ring buffer).  This is the foundation on which PER,
    N-step, HER etc. are composed via wrappers.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._obs: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._next_obs: list[torch.Tensor] = []
        self._dones: list[bool] = []
        self._head = 0  # next write position

    # ── public API ──────────────────────────────────────────────────────────
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """Store a single transition."""
        data = (obs.cpu(), action.cpu(), reward, next_obs.cpu(), done)
        if len(self._obs) < self.capacity:
            self._obs.append(data[0])
            self._actions.append(data[1])
            self._rewards.append(data[2])
            self._next_obs.append(data[3])
            self._dones.append(data[4])
        else:
            self._obs[self._head] = data[0]
            self._actions[self._head] = data[1]
            self._rewards[self._head] = data[2]
            self._next_obs[self._head] = data[3]
            self._dones[self._head] = data[4]
        self._head = (self._head + 1) % self.capacity

    def sample(self, batch_size: int) -> Batch:
        """Sample a random batch of transitions."""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        return Batch(
            obs=torch.stack([self._obs[i] for i in indices]),
            actions=torch.stack([self._actions[i] for i in indices]),
            rewards=torch.tensor([self._rewards[i] for i in indices], dtype=torch.float32),
            next_obs=torch.stack([self._next_obs[i] for i in indices]),
            dones=torch.tensor([self._dones[i] for i in indices], dtype=torch.bool),
            indices=torch.tensor(indices, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self._obs)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}, capacity={self.capacity})"
