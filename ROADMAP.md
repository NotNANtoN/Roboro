# Roadmap

Single source of truth for what's done, what's next, and what's on the horizon.

See also:
- `dev.md` — technical notes (jaxtyping, beartype, etc.)
- `docs/evaluation.md` — what to plot, multi-seed reporting, algorithm comparison
- `docs/monitoring.md` — what to track in-loop (metrics tiers, Monitor protocol)

---

## Done

- [x] Core abstractions — `Batch`, `BaseEncoder`, `BaseCritic`, `BaseActor`, `BaseUpdate`
- [x] MLP encoder, discrete/continuous Q-critics, twin Q, target networks
- [x] Replay buffer with uniform sampling (efficient terminal-obs storage)
- [x] Test infrastructure — pytest (markers: unit, integration, benchmark, slow), ruff, mypy, pre-commit
- [x] **DQN / Double DQN** — discrete Q-learning, ε-greedy, hard target copy
- [x] **DDPG** — deterministic actor-critic, Polyak targets
- [x] **SAC** — squashed Gaussian, twin Q, auto α-tuning
- [x] **TD3** — DDPG + twin Q + delayed actor + target policy smoothing
- [x] **Seeding** — `set_seed()`, deterministic env/buffer/model init, reproducibility tests
- [x] Smoke tests — every algorithm runs for ~5k steps without crashing (CartPole, Acrobot, MountainCar, Pendulum)
- [x] Initial benchmarks — convergence tests on easy classic-control envs (fast + slow tiers)

---

## Up Next

### 1. Monitoring & Evaluation Infrastructure

Before claiming proper benchmark results, we need to decide and build:

- [ ] **Decide logging backend** — WandB vs TensorBoard vs both? (WandB is already a dep in `pyproject.toml`)
- [ ] **`Monitor` protocol** — implement the hook-based design from `docs/monitoring.md` (`on_step`, `on_episode_end`, `on_eval`, `close`)
- [ ] **`WandbMonitor`** — log Tier 1 signals: episode reward, eval reward, critic loss, Q-values, gradient norms
- [ ] **Eval videos** — `gymnasium.wrappers.RecordVideo` → WandB media on eval intervals
- [ ] **Core plotting utility** — `plot_learning_curves()` and `plot_comparison()` (see `docs/evaluation.md` Phase 1)
- [ ] **Multi-seed runner** — script/utility to launch N seeds and aggregate with IQM / bootstrap CI

### 2. Full Performance Benchmarks (slow tests)

The current benchmarks are quick sanity checks on easy envs. We need proper convergence tests:

- [ ] **DQN on Breakout** (Atari) — the canonical DQN benchmark; requires CNN encoder + frame stacking
  - Depends on: CNN encoder, Atari wrappers
- [ ] **SAC on HalfCheetah / Ant** (MuJoCo) — standard continuous control benchmarks
  - Depends on: `gymnasium[mujoco]` or `gymnasium-robotics`
- [ ] **TD3 on HalfCheetah** — verify TD3 matches published results
- [ ] **DQN on CartPole** — already exists, but run with 5+ seeds and proper CI reporting
- [ ] **Multi-seed regression suite** — automated pass/fail against known score thresholds (IQM over 5 seeds)

### 3. TD3 Full Validation

TD3 is implemented and smoke-tested, but needs full performance validation:

- [ ] Run TD3 benchmark on Pendulum (50k steps) — verify convergence to ≥ −400
- [ ] Run TD3 on a harder continuous env (HalfCheetah when available)
- [ ] Compare TD3 vs DDPG vs SAC on Pendulum with matched hyperparams (multi-seed)

### 4. Encoder Integration

`MLPEncoder` exists but is never used — actors/critics build their own MLPs internally.

- [ ] **Wire encoder into critics/actors** — shared encoder trunk → actor head / critic head
- [ ] **CNN encoder** — for pixel observations (Atari, DMC pixels)
- [ ] **Encoder freezing/fine-tuning** — pre-trained encoders, stop-gradient options

### 5. WandB Integration (full)

Beyond the Monitor, integrate WandB into the training recipes:

- [ ] Hyperparameter logging (full config dict)
- [ ] Model checkpointing to WandB artifacts
- [ ] Eval video logging
- [ ] Comparison tables (algo × env × seed)

### 6. Findings from Autoresearch (March 2026)

100 autonomous experiments across CartPole+Pendulum and Hopper+Walker2d surfaced concrete improvements and process gaps. See `autoresearch/BLOG.md` for the full write-up.

**Library improvements to integrate:**

- [ ] **Orthogonal init option in `NetworkCfg`** — biggest single win in the MuJoCo campaign (+30% score). Should be `init: str = "kaiming"  # "kaiming" | "orthogonal"` with configurable gain
- [ ] **Expose `use_layer_norm` in presets** — LayerNorm was critical for stable UTD≥1 training (XQC-style). SAC/TD3 presets should default to `use_layer_norm=True`
- [ ] **Fast replay buffer alternative** — the terminal-obs optimization in `ReplayBuffer` adds overhead; a simpler explicit-storage buffer was 170s faster over 200k samples. Consider offering both or benchmarking the tradeoff
- [ ] **`max_grad_norm` for SAC/TD3** — already in DQN update but missing from continuous algorithms. Needed for stability experiments
- [ ] **UTD ratio as a `TrainCfg` parameter** — `train_freq` only supports UTD≤1. Add a `utd_ratio: int = 1` field for multiple gradient steps per env step
- [ ] **Delayed actor updates in SAC** — `actor_delay` is in TD3 config but not SAC. The autoresearch campaign showed `actor_delay=2` consistently helps SAC too

**Process improvements for autoresearch v2:**

- [ ] **Per-step metric logging from experiment 1** — critic loss, actor loss, alpha, Q-values, episode returns, gradient norms to CSV. The agent ran blind for 60 experiments
- [ ] **Auto-diagnostics between runs** — plot learning curves after each experiment, detect divergence/plateau before the full run completes
- [ ] **Failure mode clustering** — track *why* experiments failed (timeout, divergence, capacity, etc.) to avoid repeating the same class of mistake
- [ ] **Causal ablations** — when something works, automatically run single-factor ablations to confirm which component mattered
- [ ] **GPU time budget** — CPU bottleneck prevented UTD>1 entirely. On GPU, UTD=4-20 is the proven path to sample efficiency (XQC, REDQ, DroQ)

---

## Backlog

### Algorithms

- [ ] **PER** (Prioritized Experience Replay) — TD-error-proportional sampling *(tested in autoresearch: hurt SAC due to interaction with entropy tuning; may work better for DQN)*
- [ ] **N-step returns** — multi-step TD targets in the replay buffer *(tested in autoresearch: amplifies Q-value bias at low step budgets; helps Walker2d but hurts Hopper)*
- [ ] **Dueling DQN** — separate value/advantage streams
- [ ] **NoisyNets** — learned exploration (replacing ε-greedy)
- [ ] **CQL** (Conservative Q-Learning) — offline RL, conservative Q penalty
- [ ] **IQL** (Implicit Q-Learning) — offline RL, expectile regression
- [ ] **TD3+BC** — offline RL, TD3 with behavioral cloning regularization
- [ ] **CrossQ** — SAC without target networks (BatchNorm in Q-networks) *(tested in autoresearch: hurt Walker2d, BatchNorm interacts poorly with orthogonal init; needs more investigation)*

### Model-Based RL

- [ ] **`BaseDynamics`** — abstract interface already exists (`roboro/dynamics/base.py`)
- [ ] **Deterministic MLP dynamics** — simplest 1-step world model (s, a -> s', r)
- [ ] **1-Step MCTS** — pure function for Monte Carlo Tree Search using 1-step dynamics and value network
- [ ] **MCTSActor** — wrapper to use MCTS for acting (`act()` returns action + MCTS policy/value info)
- [ ] **WorldModelUpdate** — trains 1-step dynamics model and critic simultaneously
- [ ] **Sequential MuZero** — unrolled training with BPTT and SequenceBuffer (future)
- [ ] **TD-MPC2** — deterministic latent dynamics + MPPI planning
- [ ] **Dreamer-v3** — RSSM dynamics + imagination training
- [ ] **`BasePlanner`** — abstract interface already exists (`roboro/planners/base.py`)
- [ ] **MPPI planner** — sampling-based planning
- [ ] **CEM planner** — cross-entropy method

### Exploration

- [ ] **Prediction error** — curiosity-driven (ICM-style)
- [ ] **Ensemble disagreement** — epistemic uncertainty as intrinsic reward
- [ ] **`BaseExploration`** — abstract interface already exists (`roboro/exploration/base.py`)

### Infrastructure

- [ ] **`__all__` exports** in all `__init__.py` files
- [ ] **`max_grad_norm`** for DDPG / TD3 / SAC updates (DQN already has it)
- [ ] **jaxtyping + beartype** — tensor shape annotations (see `dev.md`)
- [ ] **Hydra config YAML** — example configs for CLI usage
- [ ] **Documentation site** — algorithm decomposition tables, tutorials, API docs
- [ ] **`torch.compile` testing** — systematic benchmarking of compile benefits per algorithm

---

## File Index

| File | Purpose |
|---|---|
| `ROADMAP.md` | This file — all TODOs and planning |
| `dev.md` | Technical development notes (jaxtyping, etc.) |
| `docs/evaluation.md` | What to plot, multi-seed methodology, implementation plan |
| `docs/monitoring.md` | What to track in-loop, Monitor protocol design, tier system |
