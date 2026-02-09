"""Performance benchmarks: DQN on Acrobot-v1.

CleanRL reference (3 seeds, 500k steps): -91.54 +/- 7.20

Two tiers:
  * **Fast** (~50k steps): reward clearly above random (> -300).
  * **Slow** (~500k steps): converges to >= -100.

Via pytest::

    pytest -m "benchmark and not slow" -k acrobot -s
    pytest -m benchmark -k acrobot -s

Standalone with Hydra overrides::

    python tests/benchmarks/test_acrobot_dqn.py
    python tests/benchmarks/test_acrobot_dqn.py train.device=mps
"""

from dataclasses import replace
from typing import Any

from roboro.algorithms.discrete_q import train_discrete_q
from roboro.presets import DQN
from roboro.training.trainer import TrainResult

# ── shared logic ─────────────────────────────────────────────────────────────


def _run_dqn_acrobot(cfg: Any) -> TrainResult:
    """Run DQN on Acrobot-v1.  Accepts dataclass or Hydra DictConfig."""
    print(f"Device: {cfg.train.device}  compile={cfg.train.compile}  amp={cfg.train.use_amp}")
    result = train_discrete_q("Acrobot-v1", cfg=cfg)
    print(f"\nEval rewards: {[f'{r:.0f}' for r in result.eval_rewards]}")
    if result.eval_rewards:
        print(f"Best eval: {max(result.eval_rewards):.0f}")
    return result


# ── pytest entry points (skipped when running standalone) ────────────────────

try:
    import pytest

    @pytest.mark.benchmark
    def test_dqn_acrobot_improves(train_overrides: dict[str, Any]) -> None:
        """DQN should clearly outperform random (> -300) within 50k steps.

        Random Acrobot scores ~-500.
        """
        cfg = replace(
            DQN,
            train=replace(
                DQN.train,
                total_steps=50_000,
                warmup_steps=1_000,
                eval_interval=10_000,
                eval_episodes=10,
                **train_overrides,
            ),
            epsilon_decay_steps=25_000,
        )
        result = _run_dqn_acrobot(cfg)

        assert len(result.eval_rewards) >= 2, "Need at least 2 evals"
        best_eval = max(result.eval_rewards)
        assert best_eval > -300, (
            f"DQN didn't improve beyond random on Acrobot-v1 in 50k steps. "
            f"Best eval: {best_eval:.0f} (need > -300), all evals: {result.eval_rewards}"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_dqn_acrobot_solves(train_overrides: dict[str, Any]) -> None:
        """DQN should reach >= -100 on Acrobot-v1 within 500k steps.

        CleanRL reference: -91.54 +/- 7.20
        """
        cfg = replace(DQN, train=replace(DQN.train, **train_overrides))
        result = _run_dqn_acrobot(cfg)

        assert len(result.eval_rewards) > 0, "No evaluation was performed"
        best_eval = max(result.eval_rewards)
        assert best_eval >= -100, (
            f"DQN failed on Acrobot-v1 (need >= -100). "
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
    cs.store(name="config", node=DQN)

    @hydra.main(config_name="config", version_base=None, config_path=None)
    def main(cfg: DiscreteQCfg) -> None:
        _run_dqn_acrobot(cfg)

    main()
