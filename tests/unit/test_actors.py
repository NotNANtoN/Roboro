"""Tests for actor modules."""

import torch

from roboro.actors.deterministic import DeterministicActor
from roboro.actors.epsilon_greedy import EpsilonGreedyActor
from roboro.critics.q import DiscreteQCritic


class TestEpsilonGreedyActor:
    def test_greedy_action(self, obs_dim, n_actions, batch_size):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=0.0)
        obs = torch.randn(batch_size, obs_dim)
        actions = actor.act(obs)
        assert actions.shape == (batch_size,)
        # Should always be argmax of Q
        expected = critic(obs).argmax(dim=-1)
        assert torch.equal(actions, expected)

    def test_deterministic_flag(self, obs_dim, n_actions):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=1.0)
        obs = torch.randn(1, obs_dim)
        # With deterministic=True, should ignore epsilon
        action = actor.act(obs, deterministic=True)
        expected = critic(obs).argmax(dim=-1)
        assert torch.equal(action, expected)

    def test_exploration(self, obs_dim, n_actions):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=1.0)
        obs = torch.randn(100, obs_dim)
        actions = actor.act(obs)
        # With epsilon=1.0, should have some variety (statistical, but very safe)
        assert len(actions.unique()) > 1

    def test_forward_returns_greedy(self, obs_dim, n_actions):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actor = EpsilonGreedyActor(critic, n_actions=n_actions, epsilon=0.5)
        obs = torch.randn(4, obs_dim)
        actions, log_probs = actor(obs)
        assert actions.shape == (4,)
        assert log_probs is None


class TestDeterministicActor:
    def test_output_shape(self):
        actor = DeterministicActor(obs_dim=8, action_dim=3, hidden_dim=32, n_layers=2)
        obs = torch.randn(4, 8)
        actions = actor.act(obs)
        assert actions.shape == (4, 3)

    def test_action_bounds(self):
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

    def test_noise_adds_exploration(self):
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

    def test_forward_differentiable(self):
        actor = DeterministicActor(obs_dim=4, action_dim=2, hidden_dim=16, n_layers=2)
        obs = torch.randn(4, 4, requires_grad=True)
        actions, log_probs = actor(obs)
        actions.sum().backward()
        assert obs.grad is not None
        assert log_probs is None
