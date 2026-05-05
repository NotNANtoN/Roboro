# autoresearch — RL edition

Autonomous research loop for discovering the most sample-efficient RL algorithm with a single set of hyperparameters that generalizes across environments.

Built on top of the [Roboro](../README.md) modular RL library, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `autoresearch-mar9`). The branch must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b <tag>` from the current `autoresearch` branch.
3. **Read the in-scope files**:
   - This file (`program.md`) — your instructions.
   - `prepare.py` — fixed constants, task specs, evaluation harness, time enforcement. **Do not modify.**
   - `train.py` — the file you modify. Algorithm, architecture, hyperparameters, training loop, everything.
   - `../roboro/` — the RL library you can import from but **not modify**. Browse it for available components.
4. **Initialize results.tsv** with the header row:
   ```bash
   printf 'commit\tscore\thopper\twalker\ttime_s\tstatus\tdescription\n' > results.tsv
   ```
5. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## The challenge

A single `train.py` is evaluated on **two MuJoCo locomotion environments** with the **same core hyperparameters**:

| Env | Obs | Actions | Step budget | Max return | Difficulty |
|-----|-----|---------|-------------|------------|------------|
| Hopper-v5 | 11-dim | 3-dim [-1,1] | 100,000 | ~3500 | Medium (balance + hop) |
| Walker2d-v5 | 17-dim | 6-dim [-1,1] | 100,000 | ~5000 | Hard (balance + walk) |

The shared hyperparameters (hidden_dim, n_layers, lr, gamma, batch_size, buffer_capacity, tau, etc.) must be identical for both tasks. Both tasks use SAC — algorithm-specific wiring (init_alpha, target_entropy, etc.) is allowed to differ, but the core architecture and optimization config is shared.

**Time limit**: 10 minutes wall clock for both tasks combined. **Enforced** — `prepare.py` will raise `TimeLimitExceeded` and the process will be killed via `SIGALRM` if the limit is exceeded. Call `start_timer()` at the top of `__main__` and `check_time()` between tasks.

**Run command**: `python train.py > run.log 2>&1`

## Metric

Returns are normalized to [0, 1] and averaged:

```
hopper_norm  = clamp(hopper_return / 3500, 0, 1)
walker_norm  = clamp(walker_return / 5000, 0, 1)
score = (hopper_norm + walker_norm) / 2
```

**The goal: maximize `score`.** Higher is better (max 1.0). This rewards algorithms that are sample-efficient AND generalizable across locomotion tasks of different difficulty.

**Reference**: a random policy scores ~0.004 on Hopper, ~0.0 on Walker2d. A zero-action policy scores ~0.04 on Hopper, ~0.02 on Walker2d. Reasonable SAC with 100k steps should reach 0.05–0.15.

## Experimentation

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game.
- Import anything from the `roboro` library (`../roboro/`). Browse it for available components.
- Implement new techniques directly in train.py (layer norm, batch norm, weight norm, higher UTD ratios, n-step returns, custom training loops, learning rate schedules, etc.).
- Change the algorithm structure (e.g., add delayed actor updates, target smoothing, CrossQ-style updates).

**What you CANNOT do:**
- Modify `prepare.py`. The evaluation harness is the ground truth.
- Modify any files in `../roboro/`. Treat it as a read-only library.
- Install new packages. Only use what's in `../pyproject.toml` (including `gymnasium[mujoco]`).
- Change the step budgets or evaluation seeds (defined in `prepare.py`).
- Use different core hyperparameters for the two tasks. The shared config block must be shared.
- Exceed the 10-minute wall-clock time limit.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal results is a great outcome.

**The first run**: Always establish the baseline first by running train.py as-is.

## Research directions

The goal is maximally sample-efficient continuous control with one config. Some ideas:

**Architecture:**
- Layer normalization (critical for high UTD — XQC paper, ICLR 2026)
- Batch normalization + weight normalization
- Network width/depth tuning
- Activation function experiments (ReLU, SiLU, GELU, Mish)

**Algorithms:**
- Higher UTD ratio (more gradient steps per env step — key for sample efficiency)
- Delayed actor updates (TD3-style)
- CrossQ — SAC without target networks
- Target policy smoothing

**Replay & targets:**
- N-step returns
- Higher replay ratio
- Target network strategy (Polyak tau, hard update period)

**Optimization:**
- Learning rate schedules (cosine, warmup)
- Optimizer tuning (AdamW, weight decay)
- Gradient clipping strategy
- Larger/smaller batch sizes
- Entropy coefficient tuning (fixed vs learnable alpha)

## Output format

The training script prints:

```
---
score:              0.0850
hopper_return:      350.00
hopper_norm:        0.1000
walker_return:      175.00
walker_norm:        0.0350
training_seconds:   245.3
total_seconds:      250.1
num_params_hopper:  35970
num_params_walker:  44290
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
commit	score	hopper	walker	time_s	status	description
```

1. git commit hash (short, 7 chars)
2. aggregate score (e.g. 0.0850) — use 0.0000 for crashes
3. hopper eval_return (e.g. 350.00) — use 0.00 for crashes
4. walker eval_return (e.g. 175.00) — use 0.00 for crashes
5. total wall time in seconds
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

**How to extract values from run.log and append a row:**

```bash
# Get the commit hash
COMMIT=$(git rev-parse --short HEAD)

# Extract metrics from run.log
SCORE=$(grep "^score:" run.log | awk '{print $2}')
HOPPER=$(grep "^hopper_return:" run.log | awk '{print $2}')
WALKER=$(grep "^walker_return:" run.log | awk '{print $2}')
TIME_S=$(grep "^total_seconds:" run.log | awk '{print $2}')

# Append to results.tsv (set STATUS and DESC before this)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$SCORE" "$HOPPER" "$WALKER" "$TIME_S" "$STATUS" "$DESC" >> results.tsv
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
5. Read results: `grep "^score:\|^hopper_return:\|^walker_return:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the error.
7. Record the result in results.tsv (see "Logging results" above for the exact commands).
8. Update the progress chart: `python plot.py`
9. If score improved: **keep** — advance the branch.
10. If score is equal or worse: **discard** — `git reset --hard HEAD~1` (results.tsv and progress.png are untracked, so they survive the reset).

**Timeout**: If a run exceeds 10 minutes, it will be killed automatically by prepare.py. Treat as crash.

**Crashes**: Typo or import error? Fix and re-run. Fundamentally broken? Log crash and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the roboro source, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.
