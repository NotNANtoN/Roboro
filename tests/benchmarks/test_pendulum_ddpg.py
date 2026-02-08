"""Performance benchmark: DDPG on Pendulum-v1.

This test is NOT run by default — use ``pytest -m benchmark`` to include it.
Expected: DDPG should achieve ≥ -400 mean reward within 50k steps.
(Pendulum-v1 reward range is roughly -1600 to -100; -400 is decent.)
"""

import pytest

from roboro.algorithms.ddpg import make_ddpg


@pytest.mark.benchmark
@pytest.mark.slow
def test_ddpg_pendulum_performance():
    """DDPG should reach ≥ -400 on Pendulum-v1 within 50k steps."""
    result = make_ddpg(
        "Pendulum-v1",
        hidden_dim=256,
        n_layers=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        noise_std=0.1,
        buffer_size=100_000,
        batch_size=256,
        warmup_steps=1000,
        total_steps=50_000,
        eval_interval=5000,
        eval_episodes=10,
        seed=42,
    )

    assert len(result.eval_rewards) > 0, "No evaluation was performed"

    best_eval = max(result.eval_rewards)
    assert best_eval >= -400, (
        f"DDPG failed to reach -400 on Pendulum-v1. "
        f"Best eval reward: {best_eval:.1f}, all evals: {result.eval_rewards}"
    )
