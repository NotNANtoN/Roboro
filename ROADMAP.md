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

---

## Backlog

### Algorithms

- [ ] **PER** (Prioritized Experience Replay) — TD-error-proportional sampling
- [ ] **N-step returns** — multi-step TD targets in the replay buffer
- [ ] **Dueling DQN** — separate value/advantage streams
- [ ] **NoisyNets** — learned exploration (replacing ε-greedy)
- [ ] **CQL** (Conservative Q-Learning) — offline RL, conservative Q penalty
- [ ] **IQL** (Implicit Q-Learning) — offline RL, expectile regression
- [ ] **TD3+BC** — offline RL, TD3 with behavioral cloning regularization
- [ ] **CrossQ** — SAC without target networks (BatchNorm in Q-networks)

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
