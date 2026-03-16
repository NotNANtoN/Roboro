# autoresearch v2 — DMControl edition

Autonomous research loop for sample-efficient RL on DMControl environments, with improved diagnostics and hypothesis-driven experimentation.

Built on [Roboro](../README.md). Second campaign — lessons from the [MuJoCo campaign](BLOG.md) are baked in.

## Setup

1. **Agree on a run tag**: e.g. `autoresearch-dmc-mar13`.
2. **Create the branch**: `git checkout -b <tag>` from `main`.
3. **Read the in-scope files**:
   - This file (`program_dmc.md`) — your instructions.
   - `prepare_dmc.py` — fixed constants, task specs, evaluation harness. **Do not modify.**
   - `train_dmc.py` — the file you modify. Everything is fair game.
   - `../roboro/` — the RL library you can import from but **not modify**.
4. **Initialize results.tsv**:
   ```bash
   printf 'commit\tscore\tcheetah\thumanoid\ttime_s\tstatus\tfailure_mode\tdescription\n' > results.tsv
   ```
5. **Confirm and go.**

## The challenge

A single `train_dmc.py` is evaluated on **two DMControl locomotion environments** with **shared core hyperparameters**:

| Env | Obs | Actions | Step budget | Max return | Difficulty |
|-----|-----|---------|-------------|------------|------------|
| cheetah-run | ~17-dim | 6-dim [-1,1] | 100,000 | ~1000 | Medium (fast running) |
| humanoid-walk | ~67-dim | 21-dim [-1,1] | 100,000 | ~1000 | Hard (high-dim balance + walk) |

DMControl rewards are in [0, 1] per step. Max episode return is ~1000 (1000-step episodes).

**Time limit**: 10 minutes wall clock for both tasks combined. Enforced by `prepare_dmc.py`.

**Run command**: `python train_dmc.py > run.log 2>&1`

## Metric

```
cheetah_norm  = clamp(cheetah_return / 1000, 0, 1)
humanoid_norm = clamp(humanoid_return / 1000, 0, 1)
score = (cheetah_norm + humanoid_norm) / 2
```

**Reference**: random policy scores ~0 on both. Reasonable SAC with 100k steps should reach 0.05–0.25. Target: well above 0.30.

## Experimentation rules

**What you CAN do:**
- Modify `train_dmc.py` — architecture, optimizer, hyperparameters, training loop, everything.
- Import anything from `../roboro/`.
- Implement new techniques directly in `train_dmc.py`.

**What you CANNOT do:**
- Modify `prepare_dmc.py` or any files in `../roboro/`.
- Install new packages beyond what's in `../pyproject.toml`.
- Change step budgets or evaluation seeds.
- Use different core hyperparameters for the two tasks.
- Exceed the 10-minute wall-clock time limit.

## v2 Process Improvements

### 1. Mandatory diagnostics

After EVERY run, you MUST read the metrics CSV at `runs/{task}_metrics.csv` and diagnose:

- **Q-value trajectory**: are Q-values growing steadily, or diverging/collapsing?
- **Alpha trajectory**: is the entropy coefficient finding a stable value, or oscillating?
- **Critic loss**: is it decreasing, or plateauing early?
- **Episode count**: how many episodes completed? Short episodes = falling over.

Diagnosis informs the next experiment. No experiment without a hypothesis.

### 2. Failure classification

The results.tsv has an extra column `failure_mode`. Every discard/crash gets classified:

| Mode | Description |
|------|-------------|
| `timeout` | Exceeded 600s wall clock |
| `divergence` | Q-values or critic loss exploded |
| `capacity` | Network too small for the task |
| `exploration` | Alpha collapsed, policy stuck in local optimum |
| `instability` | Score oscillated, didn't converge |
| `regression` | Helped one env, hurt the other |
| `n/a` | For keeps |

### 3. Hypothesis-driven experiments

Every experiment MUST state a hypothesis BEFORE committing:
- **Hypothesis**: "Larger hidden dim will help humanoid because 67-dim obs needs more capacity"
- **Expected outcome**: "humanoid +20%, cheetah neutral"
- **Diagnostic to check**: "if humanoid Q-values are larger, hypothesis confirmed"

Put the hypothesis in the git commit message.

### 4. Quick screening

Use short runs (20-30k steps) to screen ideas before full 100k validation:
- Create a `quick_dmc.py` screening script (env-var configurable)
- Run 2 parallel jobs (not 4 — CPU contention)
- Only validate winners on the full `train_dmc.py`

## Output format

```
---
score:              0.1500
cheetah_return:     200.00
cheetah_norm:       0.2000
humanoid_return:    100.00
humanoid_norm:      0.1000
training_seconds:   400.0
total_seconds:      450.0
num_params_cheetah: 100000
num_params_humanoid:120000
device:             cpu
```

Extract: `grep "^score:\|^cheetah_return:\|^humanoid_return:" run.log`

## Logging results

Header (8 columns — note the extra `failure_mode`):

```
commit	score	cheetah	humanoid	time_s	status	failure_mode	description
```

Example rows:
```
abc1234	0.1500	200.00	100.00	450.0	keep	n/a	baseline SAC with ortho init
def5678	0.0000	0.00	0.00	605.0	crash	timeout	UTD=2 too slow
```

## The experiment loop

LOOP FOREVER:

1. **Diagnose**: read metrics from the last run. What went well? What's the bottleneck?
2. **Hypothesize**: form a specific hypothesis for the next change.
3. **Modify** `train_dmc.py`. Put the hypothesis in the commit message.
4. `git commit` the change.
5. **Run**: `python train_dmc.py > run.log 2>&1`
6. **Read results**: `grep "^score:\|^cheetah_return:\|^humanoid_return:" run.log`
7. If empty → crash. Read `tail -n 50 run.log` for the error.
8. **Read diagnostics**: examine `runs/cheetah_metrics.csv` and `runs/humanoid_metrics.csv`.
9. **Classify** the result (keep/discard/crash + failure_mode).
10. **Record** in results.tsv.
11. **Update chart**: `python plot.py results.tsv progress.png`
12. If score improved → **keep**.
13. If equal or worse → **discard** (`git reset --hard HEAD~1`).

**NEVER STOP.** The human will interrupt when done.

## Research directions

Starting from the MuJoCo campaign's best recipe (LN + ortho init + fast buffer + delayed actor):

**High priority (proven in MuJoCo):**
- Layer normalization (already enabled)
- Orthogonal initialization (already enabled)
- Network width/depth tuning (humanoid may need wider/deeper)
- Delayed actor updates

**Medium priority (worth testing on DMC):**
- Higher UTD ratios (if time allows — benchmark per-step cost first!)
- Separate actor/critic hidden dims (humanoid's 21D action needs expressive actor)
- N-step returns (hurt MuJoCo Hopper but may help DMC's denser rewards)
- Gradient clipping on critic

**Exploratory:**
- CrossQ (BatchNorm critics, no target net — mixed results in MuJoCo)
- Action repeat / frame skip
- Reward scaling (DMC rewards already in [0,1])
- Learning rate schedules
- Weight decay / regularization

**From MuJoCo lessons — things that consistently hurt:**
- Observation/reward normalization with running stats (non-stationary buffer)
- Huber/symlog loss (wrong gradient scale)
- Cosine LR decay (RL needs constant adaptation)
- Average Q for actor (overestimation)
- Critic warm-start / delayed actor start (must co-evolve)
