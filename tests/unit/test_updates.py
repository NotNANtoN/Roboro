"""Tests for update rules (DQN, DDPG)."""

import torch

from roboro.actors.deterministic import DeterministicActor
from roboro.core.types import Batch
from roboro.critics.q import ContinuousQCritic, DiscreteQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.ddpg import DDPGUpdate
from roboro.updates.dqn import DQNUpdate


def _make_discrete_batch(obs_dim, n_actions, batch_size=16):
    """Helper: fake batch with integer actions (for DQN)."""
    return Batch(
        obs=torch.randn(batch_size, obs_dim),
        actions=torch.randint(0, n_actions, (batch_size,)).float(),
        rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, obs_dim),
        dones=torch.zeros(batch_size, dtype=torch.bool),
    )


def _make_continuous_batch(obs_dim, action_dim, batch_size=16):
    """Helper: fake batch with continuous actions (for DDPG)."""
    return Batch(
        obs=torch.randn(batch_size, obs_dim),
        actions=torch.randn(batch_size, action_dim),
        rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, obs_dim),
        dones=torch.zeros(batch_size, dtype=torch.bool),
    )


class TestDQNUpdate:
    def test_update_reduces_loss(self):
        obs_dim, n_actions = 4, 2
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions, hidden_dim=32)
        target = TargetNetwork(critic, mode="hard", hard_update_period=100)
        update = DQNUpdate(critic, target, lr=1e-2, gamma=0.99)

        batch = _make_discrete_batch(obs_dim, n_actions=n_actions, batch_size=32)
        # Run a few steps and check it doesn't crash
        results = [update.update(batch, step=i) for i in range(5)]
        assert all(r.loss >= 0 for r in results)
        assert "q_mean" in results[0].metrics

    def test_target_updates(self):
        obs_dim, n_actions = 4, 2
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions, hidden_dim=16)
        target = TargetNetwork(critic, mode="polyak", tau=0.1)
        update = DQNUpdate(critic, target, lr=1e-3)

        batch = _make_discrete_batch(obs_dim, n_actions=n_actions)
        # Get initial target params
        initial_params = [p.clone() for p in target.target.parameters()]
        update.update(batch, step=1)
        # After update, target should have changed (polyak with tau=0.1)
        for ip, cp in zip(initial_params, target.target.parameters(), strict=True):
            assert not torch.equal(ip, cp)


class TestDDPGUpdate:
    def test_update_runs(self):
        obs_dim, action_dim = 8, 2
        actor = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            n_layers=2,
        )
        actor_target = TargetNetwork(actor, mode="polyak", tau=0.005)
        critic = ContinuousQCritic(
            feature_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
        )
        critic_target = TargetNetwork(critic, mode="polyak", tau=0.005)

        update = DDPGUpdate(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            actor_lr=1e-3,
            critic_lr=1e-3,
        )

        batch = _make_continuous_batch(obs_dim, action_dim=action_dim)
        result = update.update(batch, step=1)
        assert result.loss >= 0
        assert "actor_loss" in result.metrics
        assert "critic_loss" in result.metrics

    def test_ddpg_critic_improves(self):
        obs_dim, action_dim = 4, 1
        actor = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
            n_layers=2,
        )
        actor_target = TargetNetwork(actor, mode="polyak", tau=0.01)
        critic = ContinuousQCritic(
            feature_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=32,
        )
        critic_target = TargetNetwork(critic, mode="polyak", tau=0.01)

        update = DDPGUpdate(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            actor_lr=1e-3,
            critic_lr=1e-2,
        )

        # Use a consistent batch with known rewards
        batch = _make_continuous_batch(obs_dim, action_dim=action_dim, batch_size=64)
        batch.rewards = torch.ones(64)  # constant reward
        batch.dones = torch.ones(64, dtype=torch.bool)  # terminal (target = reward only)

        # Train a few steps — critic loss should decrease on this simple signal
        losses = []
        for i in range(20):
            result = update.update(batch, step=i)
            losses.append(result.metrics["critic_loss"])

        # Loss should generally decrease (last few < first few)
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
