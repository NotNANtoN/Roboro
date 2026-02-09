"""Performance benchmarks: DQN on MountainCar-v0.

CleanRL reference (3 seeds, 500k steps): -194.95 +/- 8.48

MountainCar is a sparse-reward environment — the agent gets -1 per step
and 0 only when reaching the goal flag.  This makes it a hard exploration
problem for DQN; even CleanRL barely beats the -200 timeout floor.

Only the **slow** tier (500k steps) is meaningful here — short runs show
no learning because the agent never stumbles on the goal early enough.

Via pytest::

    pytest -m "benchmark and slow" -k mountaincar -s

Standalone with Hydra overrides::

    python tests/benchmarks/test_mountaincar_dqn.py
    python tests/benchmarks/test_mountaincar_dqn.py train.device=mps
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from roboro.algorithms.discrete_q import train_discrete_q
from roboro.presets import DQN
from roboro.training.trainer import TrainResult

# ── shared logic ─────────────────────────────────────────────────────────────


def _run_dqn_mountaincar(cfg: Any) -> TrainResult:
    """Run DQN on MountainCar-v0.  Accepts dataclass or Hydra DictConfig."""
    print(f"Device: {cfg.train.device}  compile={cfg.train.compile}  amp={cfg.train.use_amp}")
    result = train_discrete_q("MountainCar-v0", cfg=cfg)
    print(f"\nEval rewards: {[f'{r:.0f}' for r in result.eval_rewards]}")
    if result.eval_rewards:
        print(f"Best eval: {max(result.eval_rewards):.0f}")
    return result


# ── pytest entry points (skipped when running standalone) ────────────────────

try:
    import pytest

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_dqn_mountaincar_solves(train_overrides: dict[str, Any]) -> None:
        """DQN should reach >= -200 on MountainCar-v0 within 500k steps.

        CleanRL reference: -194.95 +/- 8.48
        Note: no fast tier — sparse rewards mean short runs show zero learning.
        """
        cfg = replace(DQN, train=replace(DQN.train, **train_overrides))
        result = _run_dqn_mountaincar(cfg)

        assert len(result.eval_rewards) > 0, "No evaluation was performed"
        best_eval = max(result.eval_rewards)
        assert best_eval >= -200, (
            f"DQN failed on MountainCar-v0 (need >= -200). "
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
        _run_dqn_mountaincar(cfg)

    main()
