"""Tests for actor modules."""

import torch

from roboro.actors.deterministic import DeterministicActor
from roboro.actors.epsilon_greedy import EpsilonGreedyActor
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.critics.q import DiscreteQCritic


class TestEpsilonGreedyActor:
    def test_greedy_action(self, obs_dim: int, n_actions: int, batch_size: int) -> None:
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=0.0)
        obs = torch.randn(batch_size, obs_dim)
        actions = actor.act(obs)
        assert actions.shape == (batch_size,)
        # Should always be argmax of Q
        expected = critic(obs).argmax(dim=-1)
        assert torch.equal(actions, expected)

    def test_deterministic_flag(self, obs_dim: int, n_actions: int) -> None:
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=1.0)
        obs = torch.randn(1, obs_dim)
        # With deterministic=True, should ignore epsilon
        action = actor.act(obs, deterministic=True)
        expected = critic(obs).argmax(dim=-1)
        assert torch.equal(action, expected)

    def test_exploration(self, obs_dim: int, n_actions: int) -> None:
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=1.0)
        obs = torch.randn(100, obs_dim)
        actions = actor.act(obs)
        # With epsilon=1.0, should have some variety (statistical, but very safe)
        assert len(actions.unique()) > 1

    def test_forward_returns_greedy(self, obs_dim: int, n_actions: int) -> None:
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=0.5)
        obs = torch.randn(4, obs_dim)
        actions, log_probs = actor(obs)
        assert actions.shape == (4,)
        assert log_probs is None


class TestDeterministicActor:
    def test_output_shape(self) -> None:
        actor = DeterministicActor(obs_dim=8, action_dim=3, hidden_dim=32, n_layers=2)
        obs = torch.randn(4, 8)
        actions = actor.act(obs)
        assert actions.shape == (4, 3)

    def test_action_bounds(self) -> None:
        actor = DeterministicActor(
            obs_dim=8,
            action_dim=2,
            action_low=-2.0,
            action_high=2.0,
            hidden_dim=32,
            n_layers=2,
            noise_std=0.0,
        )
        obs = torch.randn(100, 8)
        actions = actor.act(obs, deterministic=True)
        assert actions.min() >= -2.0 - 1e-6
        assert actions.max() <= 2.0 + 1e-6

    def test_noise_adds_exploration(self) -> None:
        actor = DeterministicActor(
            obs_dim=4,
            action_dim=2,
            noise_std=0.5,
            hidden_dim=16,
            n_layers=2,
        )
        obs = torch.randn(1, 4)
        # Deterministic should be consistent
        a1 = actor.act(obs, deterministic=True)
        a2 = actor.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)
        # With noise, should sometimes differ
        noisy = [actor.act(obs, deterministic=False) for _ in range(20)]
        vals = torch.stack(noisy)
        assert vals.std() > 0.0  # noise introduces variance

    def test_forward_differentiable(self) -> None:
        actor = DeterministicActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(4, 4, requires_grad=True)
        actions, log_probs = actor(obs)
        actions.sum().backward()
        assert obs.grad is not None
        assert log_probs is None


class TestSquashedGaussianActor:
    def test_output_shape(self) -> None:
        actor = SquashedGaussianActor(obs_dim=8, action_dim=3, hidden_dim=32, n_layers=2)
        obs = torch.randn(4, 8)
        actions = actor.act(obs)
        assert actions.shape == (4, 3)

    def test_forward_returns_log_probs(self) -> None:
        actor = SquashedGaussianActor(obs_dim=8, action_dim=3, hidden_dim=32, n_layers=2)
        obs = torch.randn(4, 8)
        actions, log_probs = actor(obs)
        assert actions.shape == (4, 3)
        assert log_probs is not None
        assert log_probs.shape == (4,)

    def test_action_bounds(self) -> None:
        actor = SquashedGaussianActor(
            obs_dim=8,
            action_dim=2,
            action_low=-2.0,
            action_high=2.0,
            hidden_dim=32,
            n_layers=2,
        )
        obs = torch.randn(200, 8)
        actions = actor.act(obs)
        assert actions.min() >= -2.0 - 1e-6
        assert actions.max() <= 2.0 + 1e-6

    def test_deterministic_is_consistent(self) -> None:
        actor = SquashedGaussianActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(1, 4)
        a1 = actor.act(obs, deterministic=True)
        a2 = actor.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_stochastic_has_variance(self) -> None:
        actor = SquashedGaussianActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(1, 4)
        samples = torch.stack([actor.act(obs) for _ in range(50)])
        assert samples.std() > 0.0, "Stochastic sampling should have variance"

    def test_forward_differentiable(self) -> None:
        actor = SquashedGaussianActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(4, 4, requires_grad=True)
        actions, log_probs = actor(obs)
        # Both action and log_prob must be differentiable (reparameterization)
        loss = actions.sum() + log_probs.sum()
        loss.backward()
        assert obs.grad is not None

    def test_log_probs_finite(self) -> None:
        """Log-probs should be finite (no NaN from tanh saturation)."""
        actor = SquashedGaussianActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(64, 4)
        _, log_probs = actor(obs)
        assert torch.isfinite(log_probs).all()
