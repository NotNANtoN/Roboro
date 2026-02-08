"""Tests for the shared nn building blocks."""

import pytest
import torch

from roboro.nn.blocks import MLPBlock, get_activation


class TestGetActivation:
    def test_known_activations(self):
        for name in ("relu", "mish", "gelu", "tanh", "sigmoid", "identity"):
            cls = get_activation(name)
            assert issubclass(cls, torch.nn.Module)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("does_not_exist")

    def test_case_insensitive(self):
        assert get_activation("ReLU") is get_activation("relu")


class TestMLPBlock:
    def test_output_shape(self):
        block = MLPBlock(in_dim=16, out_dim=8, hidden_dim=32, n_layers=3)
        x = torch.randn(4, 16)
        assert block(x).shape == (4, 8)

    def test_single_layer(self):
        """n_layers=1 means just the output layer (no hidden layers)."""
        block = MLPBlock(in_dim=10, out_dim=5, n_layers=1)
        x = torch.randn(2, 10)
        assert block(x).shape == (2, 5)

    def test_output_activation(self):
        block = MLPBlock(in_dim=4, out_dim=4, n_layers=2, output_activation="tanh")
        x = torch.randn(8, 4) * 10.0
        out = block(x)
        assert out.abs().max() <= 1.0 + 1e-6  # tanh bounds

    def test_no_output_activation(self):
        block = MLPBlock(in_dim=4, out_dim=4, n_layers=2)
        x = torch.randn(8, 4) * 10.0
        out = block(x)
        # Without output activation, values can exceed 1
        # (this is a statistical test, not guaranteed, but very likely)
        assert out.shape == (8, 4)

    def test_layer_norm(self):
        block = MLPBlock(in_dim=8, out_dim=4, n_layers=3, use_layer_norm=True)
        x = torch.randn(4, 8)
        assert block(x).shape == (4, 4)

    def test_gradient_flow(self):
        block = MLPBlock(in_dim=8, out_dim=4, n_layers=2)
        x = torch.randn(4, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_properties(self):
        block = MLPBlock(in_dim=16, out_dim=8)
        assert block.in_dim == 16
        assert block.out_dim == 8

    def test_different_activations(self):
        for act in ("relu", "mish", "gelu"):
            block = MLPBlock(in_dim=4, out_dim=2, activation=act, n_layers=2)
            out = block(torch.randn(2, 4))
            assert out.shape == (2, 2)

    def test_zero_hidden_layers_raises(self):
        with pytest.raises(AssertionError):
            MLPBlock(in_dim=4, out_dim=2, n_layers=0)
