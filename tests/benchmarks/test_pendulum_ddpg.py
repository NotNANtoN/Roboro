"""Performance benchmarks: DDPG on Pendulum-v1.

Two tiers:
  * **Fast** (~20k steps, ~15s): reward improves beyond random start.
  * **Slow** (~30k steps, ~60s): converges to >= -400.

Via pytest::

    pytest -m "benchmark and not slow" -k pendulum -s   # fast only
    pytest -m benchmark -k pendulum -s                   # both tiers

Standalone with Hydra overrides::

    python tests/benchmarks/test_pendulum_ddpg.py
    python tests/benchmarks/test_pendulum_ddpg.py train.device=mps critic_lr=3e-4
"""

from dataclasses import replace
from typing import Any

from roboro.algorithms.continuous_ac import train_continuous_ac
from roboro.presets import DDPG
from roboro.training.trainer import TrainResult

# ── shared logic ─────────────────────────────────────────────────────────────


def _run_ddpg_pendulum(cfg: Any) -> TrainResult:
    """Run DDPG on Pendulum-v1.  Accepts dataclass or Hydra DictConfig."""
    print(f"Device: {cfg.train.device}  compile={cfg.train.compile}  amp={cfg.train.use_amp}")
    result = train_continuous_ac("Pendulum-v1", cfg=cfg)
    print(f"\nEval rewards: {[f'{r:.0f}' for r in result.eval_rewards]}")
    if result.eval_rewards:
        print(f"Best eval: {max(result.eval_rewards):.0f}")
    return result


# ── pytest entry points (skipped when running standalone) ────────────────────

try:
    import pytest

    @pytest.mark.benchmark
    def test_ddpg_pendulum_improves(train_overrides: dict[str, Any]) -> None:
        """DDPG should improve beyond random (> -1200) within 20k steps.

        Random Pendulum scores ~-1200 to -1500.  A working DDPG should be
        clearly better by 20k steps.
        """
        cfg = replace(
            DDPG,
            train=replace(
                DDPG.train,
                total_steps=20_000,
                eval_interval=5_000,
                eval_episodes=5,
                **train_overrides,
            ),
        )
        result = _run_ddpg_pendulum(cfg)

        assert len(result.eval_rewards) >= 2, "Need at least 2 evals"
        best_eval = max(result.eval_rewards)
        assert best_eval > -1200, (
            f"DDPG didn't improve beyond random on Pendulum-v1 in 20k steps. "
            f"Best eval: {best_eval:.0f} (need > -1200), all evals: {result.eval_rewards}"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_ddpg_pendulum_solves(train_overrides: dict[str, Any]) -> None:
        """DDPG should reach >= -400 on Pendulum-v1 within 30k steps."""
        cfg = replace(DDPG, train=replace(DDPG.train, **train_overrides))
        result = _run_ddpg_pendulum(cfg)

        assert len(result.eval_rewards) > 0, "No evaluation was performed"
        best_eval = max(result.eval_rewards)
        assert best_eval >= -400, (
            f"DDPG failed to reach -400 on Pendulum-v1. "
            f"Best eval: {best_eval:.1f}, all evals: {result.eval_rewards}"
        )

except ImportError:
    pass


# ── standalone entry point (Hydra) ──────────────────────────────────────────

if __name__ == "__main__":
    import hydra
    from hydra.core.config_store import ConfigStore

    from roboro.core.config import ContinuousActorCriticCfg

    cs = ConfigStore.instance()
    cs.store(name="config", node=DDPG)

    @hydra.main(config_name="config", version_base=None, config_path=None)
    def main(cfg: ContinuousActorCriticCfg) -> None:
        _run_ddpg_pendulum(cfg)

    main()
