# autoresearch — RL edition

Autonomous research loop for discovering the most sample-efficient RL algorithm with a single set of hyperparameters that generalizes across environments.

Built on top of the [Roboro](../README.md) modular RL library, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `autoresearch-mar9`). The branch must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b <tag>` from the current `autoresearch` branch.
3. **Read the in-scope files**:
   - This file (`program.md`) — your instructions.
   - `prepare.py` — fixed constants, task specs, evaluation harness. **Do not modify.**
   - `train.py` — the file you modify. Algorithm, architecture, hyperparameters, training loop, everything.
   - `../roboro/` — the RL library you can import from but **not modify**. Browse it for available components.
4. **Initialize results.tsv** with the header row:
   ```bash
   printf 'commit\tscore\tcartpole\tpendulum\ttime_s\tstatus\tdescription\n' > results.tsv
   ```
5. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## The challenge

A single `train.py` is evaluated on **two environments** with the **same core hyperparameters**:

| Env | Type | Obs | Actions | Step budget | Max return |
|-----|------|-----|---------|-------------|------------|
| CartPole-v1 | Discrete | 4-dim | 2 | 50,000 | 500 |
| Pendulum-v1 | Continuous | 3-dim | 1-dim | 50,000 | ~0 |

The shared hyperparameters (hidden_dim, n_layers, lr, gamma, batch_size, buffer_capacity, tau, etc.) must be identical for both tasks. Algorithm-specific wiring (epsilon schedule for DQN, entropy tuning for SAC) is allowed to differ, but the core architecture and optimization config is shared.

**Time limit**: 5 minutes wall clock for both tasks combined.

**Run command**: `python train.py > run.log 2>&1`

## Metric

Returns are normalized to [0, 1] and averaged:

```
cartpole_norm = cartpole_return / 500
pendulum_norm = (pendulum_return + 1600) / 1600
score = (cartpole_norm + pendulum_norm) / 2
```

**The goal: maximize `score`.** Higher is better (max 1.0). This rewards algorithms that are sample-efficient AND generalizable.

## Experimentation

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game.
- Import anything from the `roboro` library (`../roboro/`). Browse it for available components.
- Implement new techniques directly in train.py (PER, n-step, dueling, batch norm, weight norm, distributional critics, custom training loops, etc.).
- Change the algorithm structure entirely (e.g., replace DQN with something else for discrete, or replace SAC with something else for continuous).

**What you CANNOT do:**
- Modify `prepare.py`. The evaluation harness is the ground truth.
- Modify any files in `../roboro/`. Treat it as a read-only library.
- Install new packages. Only use what's in `../pyproject.toml`.
- Change the step budgets or evaluation seeds (defined in `prepare.py`).
- Use different core hyperparameters for the two tasks. The shared config block must be shared.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal results is a great outcome.

**The first run**: Always establish the baseline first by running train.py as-is.

## Research directions

The goal is a "super modern Rainbow" — maximally sample-efficient RL with one config. Some ideas:

**Architecture:**
- Batch normalization + weight normalization (XQC paper, ICLR 2026 — key insight: well-conditioned optimization matters more than model size)
- Layer normalization in networks
- Dueling architecture for discrete Q
- Network width/depth tuning
- Activation function experiments (ReLU, SiLU, GELU)

**Algorithms:**
- Double Q-learning (already a flag in baseline)
- Distributional RL (C51 — roboro has support)
- Soft Q-learning / Munchausen RL for discrete envs
- CrossQ — SAC without target networks

**Replay & targets:**
- Prioritized Experience Replay
- N-step returns
- Higher replay ratio (more gradient steps per env step)
- Target network strategy (Polyak tau, hard update period)

**Optimization:**
- Learning rate schedules (cosine, warmup)
- Optimizer tuning (AdamW, weight decay)
- Gradient clipping strategy
- Larger/smaller batch sizes

## Output format

The training script prints:

```
---
score:              0.7200
cartpole_return:    425.00
cartpole_norm:      0.8500
pendulum_return:    -645.20
pendulum_norm:      0.5969
training_seconds:   85.3
total_seconds:      87.1
num_params_q:       1562
num_params_ac:      35970
device:             cpu
```

Extract the key metric:

```
grep "^score:" run.log
```

## Logging results

When an experiment finishes, log it to `results.tsv` (tab-separated).

The TSV has a header row and 7 columns:

```
commit	score	cartpole	pendulum	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. aggregate score (e.g. 0.7200) — use 0.0000 for crashes
3. cartpole eval_return (e.g. 425.00) — use 0.00 for crashes
4. pendulum eval_return (e.g. -645.20) — use 0.00 for crashes
5. total wall time in seconds
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

**How to extract values from run.log and append a row:**

```bash
# Get the commit hash
COMMIT=$(git rev-parse --short HEAD)

# Extract metrics from run.log
SCORE=$(grep "^score:" run.log | awk '{print $2}')
CARTPOLE=$(grep "^cartpole_return:" run.log | awk '{print $2}')
PENDULUM=$(grep "^pendulum_return:" run.log | awk '{print $2}')
TIME_S=$(grep "^total_seconds:" run.log | awk '{print $2}')

# Append to results.tsv (set STATUS and DESC before this)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$SCORE" "$CARTPOLE" "$PENDULUM" "$TIME_S" "$STATUS" "$DESC" >> results.tsv
```

For a **crash**, log it as:
```bash
printf '%s\t0.0000\t0.00\t0.00\t0.0\tcrash\t%s\n' "$COMMIT" "$DESC" >> results.tsv
```

Do NOT commit results.tsv or progress.png — leave them untracked.

## The experiment loop

LOOP FOREVER:

1. Look at the git state and results so far.
2. Choose an experimental idea. Modify `train.py`.
3. `git commit` the change with a short message describing the experiment.
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^score:\|^cartpole_return:\|^pendulum_return:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the error.
7. Record the result in results.tsv (see "Logging results" above for the exact commands).
8. Update the progress chart: `python plot.py`
9. If score improved: **keep** — advance the branch.
10. If score is equal or worse: **discard** — `git reset --hard HEAD~1` (results.tsv and progress.png are untracked, so they survive the reset).

**Timeout**: If a run exceeds 5 minutes, kill it and treat as failure.

**Crashes**: Typo or import error? Fix and re-run. Fundamentally broken? Log crash and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the roboro source, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.
