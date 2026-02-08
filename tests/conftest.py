"""Shared pytest fixtures for the Roboro test suite."""

import pytest
import torch


@pytest.fixture
def device():
    """Default device for tests — always CPU for reproducibility."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def obs_dim():
    return 16


@pytest.fixture
def action_dim():
    return 4


@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def n_actions():
    """Number of discrete actions."""
    return 10


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def random_obs(batch_size, obs_dim):
    return torch.randn(batch_size, obs_dim)


@pytest.fixture
def random_continuous_actions(batch_size, action_dim):
    return torch.randn(batch_size, action_dim)


@pytest.fixture
def random_discrete_actions(batch_size, n_actions):
    return torch.randint(0, n_actions, (batch_size,))
