"""Tests for the replay buffer."""

import numpy as np
import torch

from roboro.data.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_and_len(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=(2,))
        assert len(buf) == 0

        buf.add(
            obs=np.random.randn(4),
            action=np.random.randn(2),
            reward=1.0,
            next_obs=np.random.randn(4),
            done=False,
        )
        assert len(buf) == 1

    def test_accepts_torch_tensors(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=(2,))
        buf.add(
            obs=torch.randn(4),
            action=torch.randn(2),
            reward=1.0,
            next_obs=torch.randn(4),
            done=False,
        )
        assert len(buf) == 1

    def test_capacity_limit(self) -> None:
        buf = ReplayBuffer(capacity=5, obs_shape=(1,), action_shape=(1,))
        for i in range(10):
            buf.add(
                obs=np.array([float(i)]),
                action=np.array([0.0]),
                reward=float(i),
                next_obs=np.array([float(i + 1)]),
                done=False,
            )
        assert len(buf) == 5

    def test_sample_returns_batch(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_shape=(8,), action_shape=(2,))
        # Add enough transitions so next_obs is always valid
        for _ in range(20):
            buf.add(
                obs=np.random.randn(8),
                action=np.random.randn(2),
                reward=1.0,
                next_obs=np.random.randn(8),
                done=False,
            )
        batch = buf.sample(batch_size=4)
        assert batch.obs.shape == (4, 8)
        assert batch.actions.shape == (4, 2)
        assert batch.rewards.shape == (4,)
        assert batch.next_obs.shape == (4, 8)
        assert batch.dones.shape == (4,)
        assert batch.indices is not None
        assert batch.indices.shape == (4,)

    def test_sample_clamps_to_buffer_size(self) -> None:
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=(1,))
        # Add 3 non-terminal + 1 terminal so we have valid entries
        for _ in range(3):
            buf.add(
                obs=np.random.randn(4),
                action=np.random.randn(1),
                reward=0.0,
                next_obs=np.random.randn(4),
                done=False,
            )
        # Last non-terminal is excluded, so only 2 valid indices
        batch = buf.sample(batch_size=10)
        assert batch.batch_size == 2

    def test_ring_buffer_overwrites_oldest(self) -> None:
        buf = ReplayBuffer(capacity=3, obs_shape=(1,), action_shape=(1,))
        for i in range(5):
            buf.add(
                obs=np.array([float(i)]),
                action=np.array([0.0]),
                reward=float(i),
                next_obs=np.array([float(i + 1)]),
                done=False,
            )
        # Buffer should contain transitions 2, 3, 4 (oldest 0, 1 overwritten)
        assert len(buf) == 3
        rewards = {buf._rewards[j] for j in range(3)}
        assert rewards == {2.0, 3.0, 4.0}

    def test_terminal_obs_stored_separately(self) -> None:
        """Terminal next_obs must be stored — it can't come from obs[i+1]."""
        buf = ReplayBuffer(capacity=10, obs_shape=(2,), action_shape=())
        terminal_next = np.array([99.0, 99.0])

        buf.add(obs=np.zeros(2), action=0.0, reward=1.0, next_obs=terminal_next, done=True)
        # Next add is a reset — different obs
        buf.add(obs=np.ones(2), action=1.0, reward=0.0, next_obs=np.zeros(2), done=False)

        # Sample the terminal transition — its next_obs should be terminal_next
        batch = buf.sample(batch_size=10)
        for i in range(batch.batch_size):
            if batch.dones[i]:
                np.testing.assert_array_equal(batch.next_obs[i].numpy(), terminal_next)

    def test_discrete_action_shape(self) -> None:
        """Discrete actions (scalar) should work with action_shape=()."""
        buf = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=())
        buf.add(obs=np.zeros(4), action=2, reward=0.0, next_obs=np.zeros(4), done=False)
        buf.add(obs=np.zeros(4), action=1, reward=0.0, next_obs=np.zeros(4), done=True)
        batch = buf.sample(batch_size=2)
        assert batch.actions.ndim == 1  # (B,) not (B, 1)

    def test_repr(self) -> None:
        buf = ReplayBuffer(capacity=50, obs_shape=(4,), action_shape=(2,))
        assert "ReplayBuffer" in repr(buf)
        assert "50" in repr(buf)
