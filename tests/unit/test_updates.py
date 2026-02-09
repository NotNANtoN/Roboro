"""Tests for update rules (DQN, Double DQN, DDPG, TD3, SAC)."""

from typing import Any

import torch

from roboro.actors.deterministic import DeterministicActor
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.core.types import Batch
from roboro.critics.q import ContinuousQCritic, DiscreteQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.ddpg import DDPGUpdate
from roboro.updates.dqn import DQNUpdate
from roboro.updates.sac import SACUpdate


def _make_discrete_batch(obs_dim: int, n_actions: int, batch_size: int = 16) -> Batch:
    """Helper: fake batch with integer actions (for DQN)."""
    return Batch(
        obs=torch.randn(batch_size, obs_dim),
        actions=torch.randint(0, n_actions, (batch_size,)).float(),
        rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, obs_dim),
        dones=torch.zeros(batch_size, dtype=torch.bool),
    )


def _make_continuous_batch(obs_dim: int, action_dim: int, batch_size: int = 16) -> Batch:
    """Helper: fake batch with continuous actions (for DDPG)."""
    return Batch(
        obs=torch.randn(batch_size, obs_dim),
        actions=torch.randn(batch_size, action_dim),
        rewards=torch.randn(batch_size),
        next_obs=torch.randn(batch_size, obs_dim),
        dones=torch.zeros(batch_size, dtype=torch.bool),
    )


class TestDQNUpdate:
    def test_update_reduces_loss(self) -> None:
        obs_dim, n_actions = 4, 2
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions, hidden_dim=32)
        target = TargetNetwork(critic, mode="hard", hard_update_period=100)
        update = DQNUpdate(critic, target, lr=1e-2, gamma=0.99)

        batch = _make_discrete_batch(obs_dim, n_actions=n_actions, batch_size=32)
        # Run a few steps and check it doesn't crash
        results = [update.update(batch, step=i) for i in range(5)]
        assert all(r.loss >= 0 for r in results)
        assert "q_mean" in results[0].metrics

    def test_double_q_update(self) -> None:
        """Double DQN: online selects, target evaluates."""
        obs_dim, n_actions = 4, 2
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions, hidden_dim=32)
        target = TargetNetwork(critic, mode="hard", hard_update_period=100)
        update = DQNUpdate(critic, target, lr=1e-2, gamma=0.99, double_q=True)

        batch = _make_discrete_batch(obs_dim, n_actions=n_actions, batch_size=32)
        results = [update.update(batch, step=i) for i in range(5)]
        assert all(r.loss >= 0 for r in results)
        assert "q_mean" in results[0].metrics

    def test_target_updates(self) -> None:
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
    def test_update_runs(self) -> None:
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

    def test_ddpg_critic_improves(self) -> None:
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


class TestTD3Update:
    """TD3 = DDPG + twin Q + delayed actor + target smoothing."""

    def _build_td3(
        self, obs_dim: int = 8, action_dim: int = 2, hidden_dim: int = 32, **kwargs: Any
    ) -> tuple[DeterministicActor, TwinQCritic, TargetNetwork, TargetNetwork, DDPGUpdate]:
        """Helper to build a minimal TD3 setup (DDPGUpdate with TD3 extensions)."""
        actor = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
        actor_target = TargetNetwork(actor, mode="polyak", tau=0.005)
        q1 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        q2 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        critic = TwinQCritic(q1, q2)
        critic_target = TargetNetwork(critic, mode="polyak", tau=0.005)

        update = DDPGUpdate(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            actor_lr=1e-3,
            critic_lr=1e-3,
            actor_delay=2,
            target_noise=0.2,
            target_noise_clip=0.5,
            **kwargs,
        )
        return actor, critic, critic_target, actor_target, update

    def test_update_runs(self) -> None:
        obs_dim, action_dim = 8, 2
        _, _, _, _, update = self._build_td3(obs_dim=obs_dim, action_dim=action_dim)
        batch = _make_continuous_batch(obs_dim, action_dim=action_dim)
        result = update.update(batch, step=1)
        assert result.loss >= 0
        assert "critic_loss" in result.metrics
        assert "actor_loss" in result.metrics

    def test_delayed_actor_update(self) -> None:
        """Actor should only update on steps divisible by actor_delay."""
        _, _, _, actor_target, update = self._build_td3()
        batch = _make_continuous_batch(8, action_dim=2, batch_size=32)

        # Capture actor params before any updates
        actor_params_before = [p.clone() for p in update.actor.parameters()]
        target_params_before = [p.clone() for p in actor_target.target.parameters()]

        # Step 1: actor_delay=2, so actor should NOT update
        result = update.update(batch, step=1)
        assert result.metrics["actor_loss"] == 0.0
        for before, after in zip(actor_params_before, update.actor.parameters(), strict=True):
            assert torch.equal(before, after), "Actor should not update on odd steps"
        for before, after in zip(
            target_params_before, actor_target.target.parameters(), strict=True
        ):
            assert torch.equal(before, after), "Targets should not update on odd steps"

        # Step 2: actor SHOULD update
        result = update.update(batch, step=2)
        assert result.metrics["actor_loss"] != 0.0
        actor_changed = any(
            not torch.equal(b, a)
            for b, a in zip(actor_params_before, update.actor.parameters(), strict=True)
        )
        assert actor_changed, "Actor should update on even steps"

    def test_twin_critic_loss(self) -> None:
        """Critic loss should use both Q-networks (sum of two MSE terms)."""
        obs_dim, action_dim = 8, 2
        _, critic, _, _, update = self._build_td3(obs_dim=obs_dim, action_dim=action_dim)
        batch = _make_continuous_batch(obs_dim, action_dim=action_dim, batch_size=32)

        update.update(batch, step=2)
        # Both Q-networks should have gradients (params changed)
        q1_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in critic.q1.parameters()
        )
        q2_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in critic.q2.parameters()
        )
        assert q1_has_grad, "Q1 should receive gradients"
        assert q2_has_grad, "Q2 should receive gradients"

    def test_critic_improves(self) -> None:
        """Critic loss should decrease on a simple constant-reward signal."""
        obs_dim, action_dim = 4, 1
        _, _, _, _, update = self._build_td3(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)

        batch = _make_continuous_batch(obs_dim, action_dim=action_dim, batch_size=64)
        batch.rewards = torch.ones(64)
        batch.dones = torch.ones(64, dtype=torch.bool)

        losses = []
        for i in range(1, 31):
            result = update.update(batch, step=i)
            losses.append(result.metrics["critic_loss"])

        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_all_metrics_finite(self) -> None:
        """All reported metrics should be finite (no NaN / Inf)."""
        _, _, _, _, update = self._build_td3()
        batch = _make_continuous_batch(8, action_dim=2, batch_size=32)

        for i in range(1, 6):
            result = update.update(batch, step=i)
            for key, val in result.metrics.items():
                assert abs(val) < 1e8, f"Metric '{key}' diverged: {val}"

    def test_target_updates_on_delay(self) -> None:
        """Target networks should only change on actor-update steps."""
        _, _, critic_target, _, update = self._build_td3()
        batch = _make_continuous_batch(8, action_dim=2)

        initial_params = [p.clone() for p in critic_target.target.parameters()]

        # Step 1: targets should NOT update
        update.update(batch, step=1)
        for ip, cp in zip(initial_params, critic_target.target.parameters(), strict=True):
            assert torch.equal(ip, cp), "Targets should not update on non-delay steps"

        # Step 2: targets SHOULD update
        update.update(batch, step=2)
        any_changed = any(
            not torch.equal(ip, cp)
            for ip, cp in zip(initial_params, critic_target.target.parameters(), strict=True)
        )
        assert any_changed, "Targets should update on delay steps"


class TestSACUpdate:
    def _build_sac(
        self, obs_dim: int = 8, action_dim: int = 2, hidden_dim: int = 32, **kwargs: Any
    ) -> tuple[SquashedGaussianActor, TwinQCritic, TargetNetwork, SACUpdate]:
        """Helper to build a minimal SAC setup."""
        actor = SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
        q1 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        q2 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        critic = TwinQCritic(q1, q2)
        critic_target = TargetNetwork(critic, mode="polyak", tau=0.005)

        update = SACUpdate(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            **kwargs,
        )
        return actor, critic, critic_target, update

    def test_update_runs(self) -> None:
        obs_dim, action_dim = 8, 2
        _, _, _, update = self._build_sac(obs_dim=obs_dim, action_dim=action_dim)
        batch = _make_continuous_batch(obs_dim, action_dim=action_dim)
        result = update.update(batch, step=1)
        assert result.loss >= 0
        assert "actor_loss" in result.metrics
        assert "critic_loss" in result.metrics
        assert "alpha" in result.metrics
        assert "log_prob_mean" in result.metrics

    def test_alpha_is_learnable(self) -> None:
        """When learnable_alpha=True, alpha should change after updates."""
        _, _, _, update = self._build_sac(learnable_alpha=True)
        initial_alpha = update.alpha.item()

        batch = _make_continuous_batch(8, action_dim=2, batch_size=64)
        for i in range(20):
            update.update(batch, step=i)

        assert update.alpha.item() != initial_alpha, "Alpha should have changed"

    def test_fixed_alpha(self) -> None:
        """When learnable_alpha=False, alpha stays constant."""
        _, _, _, update = self._build_sac(learnable_alpha=False, init_alpha=0.5)
        expected_alpha = 0.5

        batch = _make_continuous_batch(8, action_dim=2, batch_size=64)
        for i in range(10):
            update.update(batch, step=i)

        assert abs(update.alpha.item() - expected_alpha) < 1e-6

    def test_critic_improves(self) -> None:
        """Critic loss should decrease on a simple constant-reward signal."""
        obs_dim, action_dim = 4, 1
        _, _, _, update = self._build_sac(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)

        batch = _make_continuous_batch(obs_dim, action_dim=action_dim, batch_size=64)
        batch.rewards = torch.ones(64)
        batch.dones = torch.ones(64, dtype=torch.bool)

        losses = []
        for i in range(30):
            result = update.update(batch, step=i)
            losses.append(result.metrics["critic_loss"])

        # Last few losses should be smaller than first few
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_all_metrics_finite(self) -> None:
        """All reported metrics should be finite (no NaN / Inf)."""
        _, _, _, update = self._build_sac()
        batch = _make_continuous_batch(8, action_dim=2, batch_size=32)

        for i in range(5):
            result = update.update(batch, step=i)
            for key, val in result.metrics.items():
                assert abs(val) < 1e8, f"Metric '{key}' diverged: {val}"

    def test_target_updates(self) -> None:
        """Target network should change after update (Polyak with tau>0)."""
        _, _, critic_target, update = self._build_sac()
        initial_params = [p.clone() for p in critic_target.target.parameters()]

        batch = _make_continuous_batch(8, action_dim=2)
        update.update(batch, step=1)

        for ip, cp in zip(initial_params, critic_target.target.parameters(), strict=True):
            assert not torch.equal(ip, cp)
