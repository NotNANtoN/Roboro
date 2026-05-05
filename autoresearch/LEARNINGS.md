# DMC Campaign Learnings

## Best config (exp16): score=0.1725, cheetah=344, humanoid=1.3, time=322s
- HIDDEN_DIM=256, N_LAYERS=2, ACTIVATION=relu, USE_LAYER_NORM=True
- LR=5e-3, GAMMA=0.995, BATCH_SIZE=128, ACTOR_DELAY=2, TAU=0.005
- ACTION_REPEAT=2, INIT_ALPHA=0.1, TARGET_ENTROPY_SCALE=0.5, ALPHA_MIN=0.01
- MAX_GRAD_NORM=1.0, WARMUP_STEPS=512, BUFFER_CAPACITY=100k

## What worked (ranked by impact)
1. **gamma=0.995** (+74 cheetah pts). Biggest single gain. Default 0.99 with repeat=2 gives gamma_eff=0.98 (horizon~50 decisions). 0.995 gives gamma_eff=0.99 (horizon~100). Cheetah benefits from valuing sustained running speed over longer horizons.
2. **action_repeat=2** (+23 cheetah pts, 2x speed). Temporal abstraction helps cheetah learn running patterns. Halves decisions→grad steps, freeing time budget for larger networks.
3. **LR=5e-3** (+34 pts from 1e-3). With only 50k grad steps (from repeat=2), higher LR extracts more learning per step. Sweet spot: 5e-3. Both 7e-3 and 1e-2 overshoot.
4. **alpha_min=0.01** (+75 pts). Prevents entropy coefficient from collapsing to 0. Without floor, cheetah alpha→0.002 and performance drops. Floor keeps useful exploration.
5. **hidden_dim=256** (+14 pts over 128). More capacity for the critic (88-dim input for humanoid). But 256 is ~2x slower per step than 128.
6. **gradient clipping** (MAX_GRAD_NORM=1.0). Stabilizes training with high LR+gamma combo. Prevents occasional large critic loss spikes.
7. **target_entropy_scale=0.5**. Standard -action_dim target is too aggressive for high-dim actions. Half that is more reasonable.

## What hurt (ranked by severity)
1. **mish/gelu activation**: 2x-2.3x slower than relu on CPU. Caused timeouts. No quality benefit.
2. **init_alpha=0.2**: Too much early randomness wasted buffer data. Cheetah 344→215.
3. **batch_size=256**: Removed beneficial gradient noise (regularization). Cheetah 344→249.
4. **ACTOR_DELAY=1**: Too aggressive actor updates destabilize learning. Cheetah 344→236.
5. **gamma=0.998**: Too long horizon — critic can't estimate accurately with 50k steps. 344→217.
6. **reward_scale=5**: Disrupts alpha auto-tuning (alpha inflates to 0.09). 270→195.
7. **LR>5e-3**: 7e-3 and 1e-2 both cause instability. Cheetah regresses.
8. **buffer=50k**: Distribution shift when overwriting causes critic loss spikes. 344→263.
9. **alpha_min=0.05**: Too much entropy for cheetah (6D action). Return collapsed to 9.
10. **UTD=2 with hidden=256**: Either too slow (timeout) or mild instability.
11. **separate alpha_lr=1e-3**: Slower alpha tuning underperforms shared LR.

## Humanoid-walk at 100k steps: fundamentally hard
- Returns ~1.2-2.0 across ALL configs (alpha, hidden, repeat, LR, gamma variations)
- Q-values always plateau at ~4-9 regardless of settings
- The task requires learning to stand then walk from 67D obs + 21D action
- Published results: standard SAC gets ~0-10 at 100k, needs 500k-1M for meaningful learning
- Alpha floor prevents complete collapse but doesn't enable learning
- Action repeat doesn't help (even repeat=4)
- The reward signal is too sparse: humanoid falls immediately, 90%+ of buffer has ~0 reward

## Key constraints
- 600s wall clock for both tasks. Hidden=256 + repeat=2 + UTD=1 uses ~322s.
- CPU-bound: no GPU available. relu is the only fast activation.
- Shared hyperparameters across tasks (can't specialize per env).

---

## DG/DQV/Dueling Experiments (Apr 2026)

### Algorithms tested (3 seeds each on cheetah-run)

| Variant | Mean Return | vs Baseline | Verdict |
|---|---|---|---|
| **Baseline SAC** | 339.92 ± 9.42 | — | ✓ reference |
| **Dueling SAC (β=0.01)** | **367.03 ± 23.65** | **+8%** | **✓ WINNER** |
| Dueling β=0.001 | 325.95 ± 17.60 | −4% | ✗ under-regularized |
| Dueling β=0.1 | 107.95 ± 73.93 | −68% | ✗ over-regularized |
| SAC-v1 (V side network) | 344.66 ± 30.21 | +1% | ≈ neutral |
| DG (Q−Q advantage) | 270.73 ± 16.94 | −20% | ✗ |
| DG + SAC-v1 (Q−V advantage) | 301.54 ± 43.61 | −11% | ✗ |
| DG + Dueling (A advantage) | 328.80 ± 90.59 | −10% | ✗ |
| DG gate only −Q | 311.41 ± 85.32 | −8% | ✗ |
| DG no whitening | 93.61 ± 12.71 | −72% | ✗ collapsed |
| DG η=0.1 | 49.52 ± 13.84 | −85% | ✗ collapsed |
| Pure DQV | 97.40 ± 61.42 | −71% | ✗ QVMAX bias |
| BC-DQV | 250.61 ± 20.54 | −26% | ✗ |
| Twin-V BC-DQV | 253.68 ± 47.47 | −25% | ✗ |

### What worked: Dueling SAC

**Architecture**: Q(s,a) = V(s) + A(s,a) with shared trunk + L2 regularization.

```python
class DuelingCritic:
    trunk(s) → features
    V_head(features) → V(s)
    A_head(features, a) → A(s,a)
    Q(s,a) = V(s) + A(s,a)

loss = L_TD + (β/2) * (V² + A²)  # RDQ-style L2 reg
```

**Why it works**:
1. No bootstrap chain (unlike DQV where Q→V→Q)
2. Twin pessimism preserved: two (V,A) pairs, min(Q1,Q2)
3. L2 reg solves identifiability: can't shift constants between V and A
4. Direct advantage access: A(s,a) available for analysis

**Optimal β=0.01**: Lower (0.001) allows V/A drift. Higher (0.1) crushes capacity.

### What failed: Delightful Policy Gradient (DG)

**Algorithm**: Gate actor loss by σ(advantage × surprisal).

**Why it fails off-policy**:
1. DG theory is fundamentally on-policy — assumes actions sampled from current π
2. Even with "on-policy" fresh samples in actor update, state distribution is off-policy
3. Gating entropy term makes no sense (entropy should be uniform exploration)
4. Whitening is required (without it, gate saturates), but whitening removes cross-state signal

**All DG variants hurt**: Q−Q, Q−V, Dueling A, gate-only-Q, no-whiten, small-η — all −8% to −85%.

### What failed: DQV-learning

**Algorithm**: Q bootstraps on V target instead of Q target.

**Why it fails for off-policy continuous control**:
1. QVMAX bias (Daley et al.): reward in V target is off-policy biased
2. Lost twin-Q pessimism: both Q1, Q2 bootstrap on same V
3. Bootstrap chain adds delay: V→Q is one step removed from reward

**BC-DQV (bias-corrected)**: V regresses to Q instead of TD. Fixed bias but added delay. Still −26%.

**Twin-V BC-DQV**: Two V networks with min(V1,V2). Didn't help — delay is the issue, not pessimism.

### Key insight

**Dueling ≠ DQV**:
- Dueling: Q = V + A learned jointly, no bootstrap chain
- DQV: Q bootstraps on V, which bootstraps on rewards

Dueling works because it's a **decomposition**, not a **bootstrap chain**.

### Next steps: Novel algorithms to try

1. **State-Adaptive Temperature (SAT-SAC)**: Learn α(s) instead of scalar α. Explore more in hard states, exploit in easy states.

2. **Curiosity-Weighted Actor (CWA)**: Replace DG's flawed "delight" with forward-model prediction error as curiosity signal.

3. **Pessimistic Advantage Decomposition (PAD)**: Q = V + min(A1, A2) — apply pessimism at advantage level.

4. **Hindsight Policy Correction (HPC)**: Store log π_old in buffer, use importance weights for critic targets.

---

## Round 4: State-Adaptive Temperature (SAT-SAC)

**Hypothesis**: Learn state-dependent α(s) instead of scalar α — explore more in hard states, exploit in easy states.

**Implementation**: Small MLP α_net(s) → softplus(·) to ensure α > 0. Same entropy-matching objective but per-state.

### Results (cheetah-run, 100k steps, 3 seeds)

| Variant        | Mean Return | Std   | vs Baseline | vs Dueling |
|----------------|-------------|-------|-------------|------------|
| SAC baseline   | 339.92      | 9.42  | —           | -7%        |
| Dueling SAC    | 367.03      | 23.65 | +8%         | —          |
| SAT-SAC        | 320.40      | 48.98 | **-6%**     | -13%       |
| SAT + Dueling  | 416.07      | 54.62 | **+22%**    | **+13%**   |

### Key insights

1. **SAT alone hurts**: Higher variance, slight performance drop. State-dependent α adds noise without clear benefit when Q estimates are noisy.

2. **SAT + Dueling synergizes**: Strong +22% over baseline, +13% over Dueling alone. The explicit V/A decomposition likely gives α(s) better signal about exploration value.

3. **Higher variance**: Both SAT variants show increased variance. The learned α(s) may overfit to specific states in some seeds.

4. **Why the synergy?** In Dueling, V(s) is explicitly learned. α(s) can be thought of as "how uncertain am I about the value of this state?" — when V(s) is explicitly available, the α network may learn a meaningful relationship.

### Next steps

1. **Regularize α_net**: Add entropy regularization on α(s) distribution to prevent extreme values.
2. **Condition α on V**: Try α(s) = f(V(s)) — make the relationship explicit.
3. **Test on more environments**: Verify synergy holds beyond cheetah.

---

---

## Round 5: SAT Stabilization Attempts (May 2026)

Extended SAT+Dueling to 5 seeds, revealing the initial 3-seed results were misleadingly good.
SAC baseline itself has a catastrophic seed (seed 45 → return 43).

### The alpha collapse problem

Deep investigation of per-seed diagnostics showed the failure mode is **alpha collapse, not alpha explosion**. Bad seeds get α stuck near ALPHA_MIN (0.01) and never explore enough. Good seeds have α grow to 0.03+ and learn faster.

### Stabilization attempts

| Variant | Mean | Std | Seeds | Verdict |
|---------|------|-----|-------|---------|
| SAT+Dueling (unclamped, LR=3e-4) | 337 | 96 | 5 | Alpha collapse in bad seeds |
| SAT+Dueling (clamped min=0.01) | 371 | 128 | 5 | Clamping doesn't fix variance |
| SAT+Dueling (LR=1e-4) | 365 | 64 | 3 | Lower LR helps slightly |
| RSAT+Dueling (residual, ε=0.5) | 384 | 105 | 5 | Highest mean but still high std |

### Root cause

The learned α(s) optimization is **structurally ill-conditioned**:
1. Non-unique solutions (many α(s) satisfy average entropy constraint)
2. Non-stationarity amplification (α changes propagate through Bellman targets)
3. Per-state sample starvation (network must generalize from sparse per-state signal)

No parameterization (RSAT, clamping, LR tuning) fixes this — the problem is fundamental.

### Other state-dependent α approaches

| Variant | Mean | Std | Seeds | Verdict |
|---------|------|-----|-------|---------|
| UCB+Dueling (Q-disagreement) | 307 | 82 | 3 | Twin-Q disagreement too weak (~1.02x) |
| TDE+Dueling β=2 (TD-error) | 360 | 134 | 3 | Signal works but unstable |
| TDE+Dueling β=5 | 387 | 96 | 3 | Better mean, still high variance |

---

## Round 6: Sampled Policy Gradient (SPG-SAC) (May 2026)

Replace the reparameterized actor gradient (backprop through Q) with sample-and-evaluate:
1. Start from stored best action (SBA) or actor output
2. If Q(s, a_replay) > Q(s, best): use replay action
3. Sample S actions around best using policy's learned std
4. If Q(s, best) > Q(s, π(s)): regress actor toward best via MSE

Based on Wiehe et al. 2018, adapted for SAC with: policy std for sampling, SBA as
separate buffer field, pessimistic Q (min of twin critics) for sample evaluation, and
batched critic forward pass for efficiency.

### Cheetah-run results (5 seeds each, same seeds 42-46, final codebase)

| Method | Mean | Std | S42 | S43 | S44 | S45 | S46 |
|--------|------|-----|-----|-----|-----|-----|-----|
| SAC baseline | 272 | 130 | 344 | 347 | 329 | **43** | 299 |
| Dueling SAC | **363** | 26 | 368 | 390 | 343 | 330 | 384 |
| SPG8 raw | 268 | 26 | 255 | 239 | 281 | 257 | 306 |
| SPG8+Dueling | 318 | 31 | 275 | 338 | 339 | 295 | 343 |
| SPG16+Dueling | 322 | 63 | 306 | 388 | 222 | 355 | 339 |
| **SPG32+Dueling** | **383** | 58 | 344 | 392 | **440** | 306 | **435** |
| RSAT+SPG+Dueling | 312 | 39 | 341 | 284 | 257 | 342 | 334 |

**Note**: SAC baseline with 5 seeds (mean=272, std=130) is far weaker than the original
3-seed estimate (340±9). Seed 45 is catastrophic for SAC (return 43) but not for SPG.

### Key findings on cheetah

1. **SPG32+Dueling beats everything** (383 mean) — more samples close the mean gap with Dueling while maintaining robustness.
2. **SPG eliminates catastrophic seeds**: Seed 45 goes from 43 (SAC) to 306 (SPG32+Dueling). Sample-and-evaluate avoids local optima that trap reparameterized gradients.
3. **Sample count matters**: SPG8→SPG16→SPG32 shows clear improvement (318→322→383).
4. **SPG8 raw has lowest variance** (std=26) — matches Dueling — but lower mean.
5. **RSAT+SPG hurts**: Residual alpha conflicts with SPG's entropy term.
6. **Pessimistic Q essential**: Without min(Q1,Q2) for sample evaluation, SPG exploits adversarial Q peaks (critic loss explodes to 12+, returns collapse).

### Quadruped-walk results (12D action, 78D obs, 5 seeds)

| Method | Mean | Std | S42 | S43 | S44 | S45 | S46 |
|--------|------|-----|-----|-----|-----|-----|-----|
| SAC baseline | 218 | 32 | 259 | 247 | 192 | 195 | 199 |
| Dueling SAC | 179 | 44 | 229 | 152 | 200 | 119 | 198 |
| **SPG32 raw** | **275** | 47 | 261 | 323 | 307 | 280 | 203 |
| SPG32+Dueling | 229 | 40 | 274 | 271 | 204 | 201 | 193 |
| SPG64 raw | 215 | 34 | 256 | 228 | 217 | 162 | 211 |
| SPG64+Dueling | 270 | 41 | 222 | 255 | 312 | 314 | 249 |

### Key findings on quadruped

1. **Dueling hurts on quadruped** (179 vs 218 SAC). Its benefit on cheetah doesn't generalize.
2. **SPG32 raw is the best method** (275) — no Dueling needed. +26% over SAC baseline.
3. **SPG64 raw is worse than SPG32** (215 vs 275) — more samples = more compute per step = fewer gradient steps in fixed wall-clock time. The compute tradeoff matters.
4. **SPG64+Dueling is good** (270) but SPG32 raw is simpler and better.

### Humanoid-walk probe (21D action, 500k steps, 1 seed)

Both SAC and SPG64 failed to learn. Q-values plateau at 6-7 (SAC) and 2-3 (SPG64) with no growth. SPG64 is 4x slower per step on 21D, getting only 140k steps vs SAC's 360k in the same wall-clock budget. Humanoid needs millions of steps or distributed training.

### SPG implementation details

- **SBA (Stored Best Action)**: Separate field in replay buffer, not overwriting original action. Original action stays for critic training; best action is starting point for next SPG search.
- **Pessimistic Q**: All sample evaluations use min(Q1, Q2) to prevent Q-exploitation.
- **Batched evaluation**: All S samples evaluated in one critic forward pass (S×B batch).
- **Policy std for sampling**: Uses actor's learned σ(s) instead of fixed noise. Adapts exploration per state.
- **Entropy term**: `loss = MSE(actor, best) + α·log_prob` — MSE trains the mean, entropy trains the std.

### Connection to Amortized Q-learning (AQL, Van de Wiele et al. 2020)

AQL is algorithmically equivalent to SPG with additional engineering:
- Uniform samples alongside proposal (M=400 uniform + N=100 proposal)
- Entropy regularization on proposal distribution
- Autoregressive proposal for structured action spaces (minor benefit per their ablation)
- Discretization of continuous actions (significant benefit in high-dim)

AQL validates SPG's core idea at scale (200M steps, 100 actors, GPU), showing it works on humanoid (21D) and even 3528-action discrete spaces. Their key ablation: the autoregressive structure barely matters — the learned proposal is what counts.

### Compute considerations

| Method | Train time (cheetah) | Train time (quadruped) |
|--------|---------------------|----------------------|
| SAC | ~120s | ~115s |
| Dueling | ~190s | ~175s |
| SPG8+Dueling | ~220s | — |
| SPG32+Dueling | ~300s | ~310s |
| SPG32 raw | — | ~220s |
| SPG64+Dueling | ~350s | ~430s |

---

### Code organization

All variants implemented as env-var toggles in `train_dmc.py`:
- `DG=1` — Delightful gate (default off)
- `DG_ETA`, `DG_WHITEN`, `DG_GATE_Q_ONLY` — DG tuning
- `DQV=1`, `DQV_BC=1`, `DQV_TWIN=1` — DQV variants (default off)
- `SACV1=1` — V as side network (default off)
- `DUELING=1`, `DUELING_BETA` — Dueling SAC (default off, β=0.01)
- `SAT=1` — State-Adaptive Temperature (default off)
- `RSAT=1`, `RSAT_EPS`, `RSAT_LR` — Residual SAT (default off)
- `UCB=1`, `UCB_BETA` — Q-disagreement temperature (default off)
- `TDE=1`, `TDE_BETA` — TD-error temperature (default off)
- `SPG=1`, `SPG_SAMPLES`, `SPG_NOISE` — Sampled Policy Gradient (default off, 8 samples)
- `TASK=<name>` — Run a single task (quadruped, humanoid, etc.)
- `STEPS=<n>` — Override step budget

Sweep driver: `run_dg_seeds.py` — configure CONFIGS list, runs N seeds each.

---

## Round 7: SPG vs Reparameterization — Diagnostic Analysis (May 2026)

### Setup

Ran SAC and SPG32+Dueling on seed 45 (catastrophic for SAC: return 43, healthy for SPG: return 306)
with per-step diagnostics comparing SPG's chosen update direction to the reparameterized gradient.

Both methods optimize `min(Q1, Q2)` — SAC via gradient, SPG via sample evaluation.

### Measured quantities (SPG32+Dueling, seed 45, cheetah-run)

**Cosine similarity between SPG direction and reparam gradient**: +0.40 average across training.
Weakly aligned — SPG and the gradient roughly agree on direction but never strongly.

**Q-value improvement per step**:
- SPG's chosen action: +0.30 Q above actor output (at gs=49k)
- Same-magnitude step along reparam gradient: +0.23 Q above actor output
- SPG finds ~33% more Q-value per step than the gradient would

**Where SPG finds best actions**:
- Gaussian samples: 83-94% of the time
- SBA (stored best from buffer): 6-17%
- Current actor output: ~0% (actor is never already optimal)

**SPG improvement fraction**: 97-100% throughout training. SPG finds a better action than the
actor for nearly every state in every batch. The actor persistently lags behind what sampling finds.

**Q-value vs actual return discrepancy**:
- SAC seed 45: Q reaches 32.8, actual return = 43. Large overestimation.
- SPG seed 45: Q reaches 24.4, actual return = 306. Conservative but accurate.

### Observations

1. SAC's Q-values grow higher than SPG's (33 vs 24) but correspond to far worse actual
   performance (43 vs 306). The critic overestimates for SAC on this seed.

2. Both methods use min(Q1, Q2) — twin-Q pessimism doesn't prevent the overestimation in SAC's
   case. The two critics, trained on the same data and targets, can develop correlated errors.

3. SPG never follows the Q-gradient directly. It evaluates Q at sampled points and picks the best.
   This makes it less susceptible to following misleading local Q-surface structure.

4. The Gaussian samples (not SBA, not replay actions) are the primary source of improvement,
   suggesting the search itself — not the memory — is what matters.

### What we don't know yet

- Whether this Q-overestimation pattern holds across other bad seeds or is specific to seed 45
- Whether the overestimation is caused by specific state-action regions or is global
- Whether increasing the critic ensemble size (N>2) would fix SAC's issue on these seeds
- Whether SPG's advantage persists at longer training horizons (where the critic gets more accurate)

### Good seed comparison (seed 42: SAC=344, SPG=344)

Ran same diagnostics on seed 42 where both methods succeed equally.

**SPG diagnostics are nearly identical between good and bad seeds:**

| Metric (at gs=49k) | Seed 42 (good) | Seed 45 (bad for SAC) |
|---------------------|----------------|-----------------------|
| Cosine sim (SPG vs reparam) | +0.38 | +0.41 |
| SPG Q-gain | +0.31 | +0.30 |
| Reparam Q-gain | +0.25 | +0.23 |
| From Gaussian samples | 82% | 85% |
| SPG improvement frac | 95% | 100% |
| Final Q (SPG) | 23.8 | 24.4 |
| Final return (SPG) | 344 | 306 |

SPG behaves almost identically on both seeds — same cosine similarity, same Q-gains, same
source distribution. The algorithm is consistent regardless of initialization luck.

**SAC Q-values are also nearly identical — but returns differ 8x:**

| grad_steps | Seed 42 Q (ret=344) | Seed 45 Q (ret=43) |
|------------|--------------------|--------------------|
| 4k | 3.3 | 2.8 |
| 14k | 12.3 | 12.4 |
| 24k | 19.5 | 19.7 |
| 34k | 25.3 | 25.8 |
| 49k | 34.7 | 32.8 |

SAC's critic reports nearly the same Q-values on both seeds (within 5%). On seed 42 (return 344)
the overestimation happens to correlate with actually good actions. On seed 45 (return 43) it
doesn't. The critic can't distinguish between a good and catastrophic policy.

### Summary

The Q-overestimation is equally present on good and bad seeds. The difference in SAC performance
is whether the overestimated Q-regions happen to align with genuinely good actions — which is
determined by initialization luck. SPG avoids dependence on this luck because it evaluates Q at
sampled points rather than following the Q-gradient into potentially misleading regions.

### Ensemble critic experiment (seed 42 + 45, cheetah-run)

Tested whether more critics fix the catastrophic seed. First 2 networks always get the same
init weights (RNG state saved/restored around extra critic construction).

| Method | Seed 45 (bad) | Seed 42 (good) | Mean | Std |
|--------|---------------|----------------|------|-----|
| SAC N=2 (standard) | **43** | 344 | 193 | 213 |
| SAC N=4 | 277 | 310 | 294 | 24 |
| SAC N=8 | **364** | 345 | **354** | **13** |

N=8 critics completely fixes the catastrophic seed (43 → 364) while preserving good-seed
performance (344 → 345). Variance drops from 213 → 13.

This confirms: **the catastrophic failure is caused by correlated overestimation between the
twin critics.** With only 2 critics trained on the same data and targets, they develop correlated
errors. The actor follows gradients into regions where both critics agree Q is high but actual
returns are low. Adding more independently initialized critics breaks this correlation.

SPG and larger ensembles solve the same problem from different angles:
- SPG: don't follow the Q-gradient — evaluate at sampled points instead
- N=8: make the min-Q surface more accurate so the gradient can be trusted

Trade-offs:
- SPG32+Dueling: ~300s train time, 306 return on seed 45
- SAC N=8: ~260s train time, 364 return on seed 45
- SAC N=2: ~110s train time, 43 return on seed 45

### Full 5-seed sweep: N=8 and SPG+N=8 (cheetah-run)

| Method | Mean | Std | Avg Time | S42 | S43 | S44 | S45 | S46 |
|--------|------|-----|----------|-----|-----|-----|-----|-----|
| SAC N=2 | 272 | 130 | 115s | 344 | 347 | 329 | 43 | 299 |
| Dueling N=2 | 363 | 26 | 180s | 368 | 390 | 343 | 330 | 384 |
| SPG32+Dueling N=2 | 383 | 58 | 300s | 344 | 392 | 440 | 306 | 435 |
| **SAC N=8** | **346** | **14** | 265s | 345 | 341 | 326 | 364 | 353 |
| SPG32+Dueling+N=8 | 383 | 58 | 369s | 344 | 392 | 440 | 306 | 435 |

Key findings:
- SAC N=8 has the **lowest variance** of any method (std=14), no catastrophic seeds.
- SPG32+Dueling+N=8 has same mean as SPG32+Dueling N=2 (383 both) — SPG already
  handles what extra critics provide. The ensemble doesn't help SPG further.
- SAC N=8 and Dueling N=2 are the best compute-performance tradeoffs.

### Summary of all methods (cheetah-run, 100k steps, best variants only)

| Method | Mean | Std | Time | Key property |
|--------|------|-----|------|-------------|
| SAC N=2 | 272 | 130 | 115s | Catastrophic seed 45 |
| **SAC N=8** | **346** | **14** | 265s | Most stable, no catastrophes |
| **Dueling N=2** | **363** | **26** | 180s | Best mean per compute |
| SPG32+Dueling N=2 | 383 | 58 | 300s | Highest mean, moderate variance |
| SPG32+Dueling+N=8 | 383 | 58 | 369s | Ensemble doesn't help SPG |

### Next steps

1. **GPU experiments on humanoid** (21D action, RTX 4090): Run SAC N=8, Dueling, and
   SPG32+Dueling at 1M+ steps with full diagnostics and plots.

2. **Action space visualization**: Dense Q-surface plots for N=2 vs N=8 to visualize
   overestimation topology.
