"""Performance benchmarks: C51 (Categorical DQN) on CartPole-v1.

Two tiers:
  * **Fast** (~50k steps): reward clearly above random (> 100).
  * **Slow** (~500k steps): near-solved performance (>= 475).

Via pytest::

    pytest -m "benchmark and not slow" -k c51 -s   # fast only
    pytest -m benchmark -k c51 -s                   # both tiers
    pytest -m benchmark -k c51 -s --device mps      # on GPU
"""

from dataclasses import replace
from typing import Any

from roboro.algorithms.discrete_q import train_discrete_q
from roboro.presets import C51
from roboro.training.trainer import TrainResult

# ── shared logic ─────────────────────────────────────────────────────────────


def _run_c51_cartpole(cfg: Any) -> TrainResult:
    """Run C51 on CartPole-v1."""
    print(f"Device: {cfg.train.device}  compile={cfg.train.compile}  amp={cfg.train.use_amp}")
    result = train_discrete_q("CartPole-v1", cfg=cfg)
    print(f"\nEval rewards: {[f'{r:.0f}' for r in result.eval_rewards]}")
    if result.eval_rewards:
        print(f"Best eval: {max(result.eval_rewards):.0f}")
    return result


# ── pytest entry points (skipped when running standalone) ────────────────────

try:
    import pytest

    @pytest.mark.benchmark
    def test_c51_cartpole_improves(train_overrides: dict[str, Any]) -> None:
        """C51 should clearly outperform random (>100) within 50k steps."""
        cfg = replace(
            C51,
            train=replace(
                C51.train,
                total_steps=50_000,
                warmup_steps=1_000,
                eval_interval=10_000,
                eval_episodes=10,
                **train_overrides,
            ),
            epsilon_decay_steps=25_000,
        )
        result = _run_c51_cartpole(cfg)

        assert len(result.eval_rewards) >= 2, "Need at least 2 evals"
        best_eval = max(result.eval_rewards)
        assert best_eval > 100, (
            f"C51 didn't improve beyond random on CartPole-v1 in 50k steps. "
            f"Best eval: {best_eval:.0f} (need >100), all evals: {result.eval_rewards}"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_c51_cartpole_solves(train_overrides: dict[str, Any]) -> None:
        """C51 should solve CartPole-v1 (>= 475 avg reward) within 500k steps."""
        cfg = replace(C51, train=replace(C51.train, **train_overrides))
        result = _run_c51_cartpole(cfg)

        assert len(result.eval_rewards) > 0, "No evaluation was performed"
        best_eval = max(result.eval_rewards)
        assert best_eval >= 475, (
            f"C51 failed to solve CartPole-v1 (need >= 475). "
            f"Best eval: {best_eval:.1f}, all evals: {result.eval_rewards}"
        )

except ImportError:
    pass


# ── standalone entry point (Hydra) ──────────────────────────────────────────

if __name__ == "__main__":
    import hydra
    from hydra.core.config_store import ConfigStore

    from roboro.core.config import DiscreteQCfg

    cs = ConfigStore.instance()
    cs.store(name="config", node=C51)

    @hydra.main(config_name="config", version_base=None, config_path=None)
    def main(cfg: DiscreteQCfg) -> None:
        _run_c51_cartpole(cfg)

    main()
