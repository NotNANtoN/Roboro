"""Performance benchmarks: 1-Step Model-Based (Simplified MuZero) on CartPole-v1."""

from dataclasses import replace
from typing import Any

from roboro.algorithms.model_based import train_model_based
from roboro.presets import MUZERO_1STEP
from roboro.training.trainer import TrainResult

# ── shared logic ─────────────────────────────────────────────────────────────


def _run_muzero_cartpole(cfg: Any) -> TrainResult:
    """Run 1-Step MuZero on CartPole-v1."""
    print(f"Device: {cfg.train.device}  compile={cfg.train.compile}  amp={cfg.train.use_amp}")
    result = train_model_based("CartPole-v1", cfg=cfg)
    print(f"\nEval rewards: {[f'{r:.0f}' for r in result.eval_rewards]}")
    if result.eval_rewards:
        print(f"Best eval: {max(result.eval_rewards):.0f}")
    return result


# ── pytest entry points ──────────────────────────────────────────────────────

try:
    import pytest

    @pytest.mark.benchmark
    def test_muzero_1step_cartpole_improves(train_overrides: dict[str, Any]) -> None:
        """1-Step MuZero should clearly outperform random (>100) within 10k steps.

        Random CartPole scores ~20-30.
        Model-based methods are sample efficient, so 10k steps should be plenty
        to show learning progress.
        """
        cfg = replace(
            MUZERO_1STEP,
            train=replace(
                MUZERO_1STEP.train,
                total_steps=10_000,
                warmup_steps=1_000,
                eval_interval=2_500,
                eval_episodes=5,
                **train_overrides,
            ),
            num_simulations=15,
        )
        result = _run_muzero_cartpole(cfg)

        assert len(result.eval_rewards) >= 2, "Need at least 2 evals"
        best_eval = max(result.eval_rewards)
        assert best_eval > 100, (
            f"1-Step MuZero didn't improve beyond random on CartPole-v1 in 10k steps. "
            f"Best eval: {best_eval:.0f} (need >100), all evals: {result.eval_rewards}"
        )

except ImportError:
    pass
