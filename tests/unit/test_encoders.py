"""Tests for encoder modules."""

import torch

from roboro.encoders.mlp import MLPEncoder


class TestMLPEncoder:
    def test_output_shape(self, random_obs: torch.Tensor, obs_dim: int, hidden_dim: int) -> None:
        feature_dim = 64
        encoder = MLPEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim, feature_dim=feature_dim)
        features = encoder(random_obs)
        assert features.shape == (random_obs.shape[0], feature_dim)

    def test_feature_dim_property(self, obs_dim: int) -> None:
        encoder = MLPEncoder(obs_dim=obs_dim, feature_dim=128)
        assert encoder.feature_dim == 128

    def test_different_activations(self, random_obs: torch.Tensor, obs_dim: int) -> None:
        for act in ("relu", "mish", "gelu", "tanh"):
            encoder = MLPEncoder(obs_dim=obs_dim, activation=act)
            out = encoder(random_obs)
            assert out.shape[0] == random_obs.shape[0]

    def test_layer_norm(self, random_obs: torch.Tensor, obs_dim: int) -> None:
        encoder = MLPEncoder(obs_dim=obs_dim, use_layer_norm=True)
        out = encoder(random_obs)
        assert out.shape[0] == random_obs.shape[0]

    def test_n_layers(self, random_obs: torch.Tensor, obs_dim: int) -> None:
        for n in (2, 3, 4):
            encoder = MLPEncoder(obs_dim=obs_dim, n_layers=n)
            out = encoder(random_obs)
            assert out.shape[0] == random_obs.shape[0]

    def test_gradient_flow(self, obs_dim: int) -> None:
        encoder = MLPEncoder(obs_dim=obs_dim, feature_dim=32)
        obs = torch.randn(4, obs_dim, requires_grad=True)
        features = encoder(obs)
        loss = features.sum()
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0
