"""Tests for the replay buffer."""

import torch

from roboro.data.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

        buf.add(
            obs=torch.randn(4),
            action=torch.randn(2),
            reward=1.0,
            next_obs=torch.randn(4),
            done=False,
        )
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.add(
                obs=torch.tensor([float(i)]),
                action=torch.tensor([0.0]),
                reward=float(i),
                next_obs=torch.tensor([float(i + 1)]),
                done=False,
            )
        assert len(buf) == 5

    def test_sample_returns_batch(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.add(
                obs=torch.randn(8),
                action=torch.randn(2),
                reward=1.0,
                next_obs=torch.randn(8),
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

    def test_sample_clamps_to_buffer_size(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(3):
            buf.add(
                obs=torch.randn(4),
                action=torch.randn(1),
                reward=0.0,
                next_obs=torch.randn(4),
                done=False,
            )
        batch = buf.sample(batch_size=10)
        assert batch.batch_size == 3  # only 3 available

    def test_ring_buffer_overwrites_oldest(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(
                obs=torch.tensor([float(i)]),
                action=torch.tensor([0.0]),
                reward=float(i),
                next_obs=torch.tensor([float(i + 1)]),
                done=False,
            )
        # Buffer should contain transitions 2, 3, 4 (oldest 0, 1 overwritten)
        assert len(buf) == 3
        rewards = {buf._rewards[j] for j in range(3)}
        assert rewards == {2.0, 3.0, 4.0}

    def test_repr(self):
        buf = ReplayBuffer(capacity=50)
        assert "ReplayBuffer" in repr(buf)
        assert "50" in repr(buf)
