"""Smoke tests: verify algorithms learn at all (~5k steps, <10s each).

These catch broken gradients, shape errors, and gross wiring bugs.
They do NOT verify convergence — see ``tests/benchmarks/`` for that.

Run::

    pytest tests/integration/test_smoke.py -v
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from roboro.algorithms.continuous_ac import train_continuous_ac
from roboro.algorithms.discrete_q import train_discrete_q
from roboro.core.config import ContinuousActorCriticCfg, DiscreteQCfg
from roboro.presets import DDPG, DOUBLE_DQN, DQN, SAC

# ── shared config for speed ─────────────────────────────────────────────────

_FAST_STEPS = 5_000
_FAST_EVAL = 1_000


def _fast_train[T: (DiscreteQCfg, ContinuousActorCriticCfg)](preset: T) -> T:
    """Return a copy of *preset* with minimal training steps for smoke testing."""
    return replace(
        preset,
        train=replace(
            preset.train,
            total_steps=_FAST_STEPS,
            warmup_steps=200,
            batch_size=64,
            train_freq=4,
            eval_interval=_FAST_EVAL,
            eval_episodes=3,
            show_progress=False,
        ),
    )


def _fast_discrete(preset: DiscreteQCfg, **extra: Any) -> DiscreteQCfg:
    """_fast_train + scaled epsilon decay for discrete Q presets."""
    cfg = _fast_train(preset)
    return replace(cfg, epsilon_decay_steps=2_500, **extra)


# ── Discrete Q ──────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestDiscreteQSmoke:
    """Quick sanity checks for discrete Q-learning variants."""

    def test_dqn_runs_and_evaluates(self) -> None:
        """DQN completes training and produces eval rewards."""
        cfg = _fast_discrete(DQN)
        result = train_discrete_q("CartPole-v1", cfg=cfg)
        assert len(result.eval_rewards) >= 2
        assert len(result.episode_rewards) > 0

    def test_dqn_loss_is_finite(self) -> None:
        """DQN produces finite loss values (no NaN / Inf divergence)."""
        cfg = _fast_discrete(DQN)
        result = train_discrete_q("CartPole-v1", cfg=cfg)
        losses = [m["loss"] for m in result.metrics if "loss" in m]
        assert len(losses) > 0, "No gradient steps happened"
        assert all(
            abs(v) < 1e6 for v in losses
        ), f"Loss diverged: max={max(abs(v) for v in losses)}"

    def test_double_dqn_runs(self) -> None:
        """Double DQN completes without errors."""
        cfg = _fast_discrete(DOUBLE_DQN)
        result = train_discrete_q("CartPole-v1", cfg=cfg)
        assert len(result.eval_rewards) >= 2

    def test_acrobot(self) -> None:
        """DQN runs on Acrobot-v1 (6-dim obs, 3 discrete actions)."""
        cfg = _fast_discrete(DQN)
        result = train_discrete_q("Acrobot-v1", cfg=cfg)
        assert len(result.eval_rewards) >= 2

    def test_mountaincar(self) -> None:
        """DQN runs on MountainCar-v0 (2-dim obs, 3 discrete actions)."""
        cfg = _fast_discrete(DQN)
        result = train_discrete_q("MountainCar-v0", cfg=cfg)
        assert len(result.eval_rewards) >= 2


# ── Continuous actor-critic ─────────────────────────────────────────────────


@pytest.mark.integration
class TestContinuousACSmoke:
    """Quick sanity checks for continuous actor-critic variants."""

    def test_ddpg_runs_and_evaluates(self) -> None:
        """DDPG completes training and produces eval rewards."""
        cfg = _fast_train(DDPG)
        result = train_continuous_ac("Pendulum-v1", cfg=cfg)
        assert len(result.eval_rewards) >= 2
        assert len(result.episode_rewards) > 0

    def test_ddpg_reward_improves(self) -> None:
        """DDPG's later episodes should score better than the first ones."""
        cfg = _fast_train(DDPG)
        result = train_continuous_ac("Pendulum-v1", cfg=cfg)
        n = min(5, len(result.episode_rewards) // 3)
        if n < 2:
            pytest.skip("Not enough episodes for comparison")
        first = result.episode_rewards[:n]
        last = result.episode_rewards[-n:]
        # Pendulum: higher is better (less negative)
        assert sum(last) / len(last) > sum(first) / len(first), (
            f"DDPG didn't improve: first_avg={sum(first) / len(first):.0f}, "
            f"last_avg={sum(last) / len(last):.0f}"
        )

    def test_sac_runs_and_evaluates(self) -> None:
        """SAC completes training and produces eval rewards."""
        cfg = _fast_train(SAC)
        result = train_continuous_ac("Pendulum-v1", cfg=cfg)
        assert len(result.eval_rewards) >= 2
        assert len(result.episode_rewards) > 0

    def test_sac_loss_is_finite(self) -> None:
        """SAC produces finite loss values (no NaN / Inf divergence)."""
        cfg = _fast_train(SAC)
        result = train_continuous_ac("Pendulum-v1", cfg=cfg)
        losses = [m["loss"] for m in result.metrics if "loss" in m]
        assert len(losses) > 0, "No gradient steps happened"
        assert all(
            abs(v) < 1e6 for v in losses
        ), f"Loss diverged: max={max(abs(v) for v in losses)}"

    def test_sac_alpha_tracked(self) -> None:
        """SAC metrics include the entropy coefficient alpha."""
        cfg = _fast_train(SAC)
        result = train_continuous_ac("Pendulum-v1", cfg=cfg)
        alphas = [m["alpha"] for m in result.metrics if "alpha" in m]
        assert len(alphas) > 0, "Alpha not tracked in metrics"
        assert all(a > 0 for a in alphas), "Alpha should be positive"
