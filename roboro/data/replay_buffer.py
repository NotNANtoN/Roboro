"""Uniform replay buffer with efficient observation storage.

The key insight: in a sequential buffer, ``next_obs[i] == obs[i + 1]``
for non-terminal transitions.  We only store ``obs`` and keep a sparse
dict of terminal observations, cutting observation memory nearly in half.
"""

import numpy as np
import torch

from roboro.core.types import Batch


def _as_numpy(x: np.ndarray | torch.Tensor | float | int) -> np.ndarray:
    """Convert any input to a float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class ReplayBuffer:
    """Ring-buffer replay with efficient next-obs storage.

    Only *terminal* ``next_obs`` are stored separately; for non-terminal
    transitions, the next observation is simply ``obs[(i+1) % capacity]``.

    Args:
        capacity: maximum number of transitions.
        obs_shape: shape of a single observation, e.g. ``(4,)`` or ``(84, 84, 3)``.
        action_shape: shape of a single action.
            ``()`` for discrete (scalar), ``(action_dim,)`` for continuous.
        seed: optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        # Terminal observations: only stored at episode boundaries.
        self._terminal_obs: dict[int, np.ndarray] = {}

        self._pos = 0  # next write position
        self._size = 0  # current number of stored transitions
        self._rng = np.random.default_rng(seed)  # fast, modern NumPy RNG

    # ── public API ──────────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        action: np.ndarray | torch.Tensor | float | int,
        reward: float,
        next_obs: np.ndarray | torch.Tensor,
        done: bool,
    ) -> None:
        """Store a single transition.

        ``next_obs`` is only kept for terminal transitions (where ``done=True``).
        For non-terminal transitions the next observation is derived from the
        subsequent entry in the ring buffer.
        """
        idx = self._pos

        self._obs[idx] = _as_numpy(obs)
        self._actions[idx] = _as_numpy(action)
        self._rewards[idx] = float(reward)
        self._dones[idx] = bool(done)

        if done:
            self._terminal_obs[idx] = _as_numpy(next_obs)
        elif idx in self._terminal_obs:
            # Clean up if this slot previously held a terminal transition
            del self._terminal_obs[idx]

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample a random batch of transitions (uniform, with replacement).

        O(batch_size) — no allocations proportional to buffer size.

        With-replacement is standard for RL replay (same as SB3).  At
        batch_size=128 / buffer=50k the duplicate probability is < 0.2%.
        """
        # Exclude the most recent non-terminal entry (no valid next_obs yet)
        last = (self._pos - 1) % self._size
        exclude_last = not self._dones[last]
        pool = self._size - int(exclude_last)
        n = min(batch_size, pool)

        # O(n) uniform random integers
        raw = self._rng.integers(0, pool, size=n)

        # If we're excluding `last`, shift indices >= last to skip it
        if exclude_last:
            raw[raw >= last] += 1

        indices = raw.astype(np.int64)

        # Build next_obs: default from next slot, override for terminals
        next_indices = (indices + 1) % self._size
        next_obs = self._obs[next_indices].copy()

        # Vectorised terminal mask — only iterate the (few) terminal hits
        terminal_mask = self._dones[indices]
        if terminal_mask.any():
            for i in np.flatnonzero(terminal_mask):
                next_obs[i] = self._terminal_obs[indices[i]]

        return Batch(
            obs=torch.from_numpy(self._obs[indices]),
            actions=torch.from_numpy(self._actions[indices]),
            rewards=torch.from_numpy(self._rewards[indices]),
            next_obs=torch.from_numpy(next_obs),
            dones=torch.from_numpy(self._dones[indices]),
            indices=torch.from_numpy(indices),
        )

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}, capacity={self.capacity}, "
            f"obs_shape={self.obs_shape}, action_shape={self.action_shape})"
        )
