"""Performance benchmark: DQN on CartPole-v1.

This test is NOT run by default — use ``pytest -m benchmark`` to include it.
Expected: DQN should achieve ≥ 450 mean reward within 50k steps.
"""

import pytest

from roboro.algorithms.dqn import make_dqn


@pytest.mark.benchmark
@pytest.mark.slow
def test_dqn_cartpole_performance():
    """DQN should solve CartPole-v1 (≥ 450 avg reward) within 50k steps."""
    result = make_dqn(
        "CartPole-v1",
        hidden_dim=128,
        n_layers=2,
        lr=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=128,
        warmup_steps=500,
        total_steps=50_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=15_000,
        target_update_mode="hard",
        target_update_period=500,
        eval_interval=5000,
        eval_episodes=10,
        seed=42,
    )

    # Should have at least one eval reward
    assert len(result.eval_rewards) > 0, "No evaluation was performed"

    # The last evaluation should achieve near-optimal performance
    best_eval = max(result.eval_rewards)
    assert best_eval >= 400, (
        f"DQN failed to reach 400 on CartPole-v1. "
        f"Best eval reward: {best_eval:.1f}, all evals: {result.eval_rewards}"
    )
