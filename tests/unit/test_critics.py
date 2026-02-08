"""Tests for critic modules."""

import pytest
import torch

from roboro.critics.q import ContinuousQCritic, DiscreteQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork


class TestDiscreteQCritic:
    def test_output_shape_all_actions(self, random_obs, obs_dim, n_actions):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        q_vals = critic(random_obs)
        assert q_vals.shape == (random_obs.shape[0], n_actions)

    def test_output_shape_specific_actions(self, random_obs, obs_dim, n_actions, batch_size):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        actions = torch.randint(0, n_actions, (batch_size,))
        q_vals = critic(random_obs, actions)
        assert q_vals.shape == (batch_size,)

    def test_gradient_flow(self, obs_dim, n_actions):
        critic = DiscreteQCritic(feature_dim=obs_dim, n_actions=n_actions)
        obs = torch.randn(4, obs_dim, requires_grad=True)
        q_vals = critic(obs)
        loss = q_vals.sum()
        loss.backward()
        assert obs.grad is not None


class TestContinuousQCritic:
    def test_output_shape(self, random_obs, random_continuous_actions, obs_dim, action_dim):
        critic = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        q_vals = critic(random_obs, random_continuous_actions)
        assert q_vals.shape == (random_obs.shape[0],)

    def test_requires_actions(self, random_obs, obs_dim, action_dim):
        critic = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        with pytest.raises(ValueError, match="requires explicit actions"):
            critic(random_obs)

    def test_gradient_flow(self, obs_dim, action_dim):
        critic = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        obs = torch.randn(4, obs_dim, requires_grad=True)
        actions = torch.randn(4, action_dim, requires_grad=True)
        q_val = critic(obs, actions)
        loss = q_val.sum()
        loss.backward()
        assert obs.grad is not None
        assert actions.grad is not None


class TestTwinQCritic:
    def test_min_output(self, random_obs, random_continuous_actions, obs_dim, action_dim):
        q1 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        q2 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        twin = TwinQCritic(q1, q2)
        min_q = twin(random_obs, random_continuous_actions)
        q1_val, q2_val = twin.both(random_obs, random_continuous_actions)
        expected_min = torch.min(q1_val, q2_val)
        assert torch.allclose(min_q, expected_min)

    def test_both_returns_two(self, random_obs, random_continuous_actions, obs_dim, action_dim):
        q1 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        q2 = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        twin = TwinQCritic(q1, q2)
        v1, v2 = twin.both(random_obs, random_continuous_actions)
        assert v1.shape == v2.shape


class TestTargetNetwork:
    def test_initial_copy(self, obs_dim, action_dim):
        source = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        target = TargetNetwork(source, mode="polyak", tau=0.005)
        # Target should produce the same output as source initially
        obs = torch.randn(4, obs_dim)
        actions = torch.randn(4, action_dim)
        with torch.no_grad():
            src_out = source(obs, actions)
            tgt_out = target(obs, actions)
        assert torch.allclose(src_out, tgt_out)

    def test_polyak_update_changes_target(self, obs_dim, action_dim):
        source = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        target = TargetNetwork(source, mode="polyak", tau=0.005)

        # Modify source weights
        for p in source.parameters():
            p.data += 1.0

        obs = torch.randn(4, obs_dim)
        actions = torch.randn(4, action_dim)

        with torch.no_grad():
            before = target(obs, actions).clone()

        target.update()

        with torch.no_grad():
            after = target(obs, actions)

        # Target should have moved toward source
        assert not torch.allclose(before, after)

    def test_target_params_frozen(self, obs_dim, action_dim):
        source = ContinuousQCritic(feature_dim=obs_dim, action_dim=action_dim)
        target_net = TargetNetwork(source)
        for p in target_net.target.parameters():
            assert not p.requires_grad
