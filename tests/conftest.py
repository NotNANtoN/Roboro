"""Shared pytest fixtures and CLI options for the Roboro test suite."""

from typing import Any

import pytest
import torch

# ── CLI options ──────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Roboro-specific CLI flags for benchmark runs."""
    parser.addoption("--device", default="cpu", help="cpu | cuda | mps")
    parser.addoption("--compile", action="store_true", default=False)
    parser.addoption("--amp", action="store_true", default=False)


# ── Benchmark fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def train_overrides(request: pytest.FixtureRequest) -> dict[str, Any]:
    """CLI overrides for ``TrainCfg`` fields (device, compile, use_amp)."""
    return {
        "device": request.config.getoption("--device"),
        "compile": request.config.getoption("--compile"),
        "use_amp": request.config.getoption("--amp"),
    }


# ── Standard unit-test fixtures ─────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    """Default device for unit tests — always CPU."""
    return torch.device("cpu")


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def obs_dim() -> int:
    return 16


@pytest.fixture
def action_dim() -> int:
    return 4


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def n_actions() -> int:
    """Number of discrete actions."""
    return 10


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def random_obs(batch_size: int, obs_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, obs_dim)


@pytest.fixture
def random_continuous_actions(batch_size: int, action_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, action_dim)


@pytest.fixture
def random_discrete_actions(batch_size: int, n_actions: int) -> torch.Tensor:
    return torch.randint(0, n_actions, (batch_size,))
