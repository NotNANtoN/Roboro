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

### Code organization

All variants implemented as env-var toggles in `train_dmc.py`:
- `DG=1` — Delightful gate (default off)
- `DG_ETA`, `DG_WHITEN`, `DG_GATE_Q_ONLY` — DG tuning
- `DQV=1`, `DQV_BC=1`, `DQV_TWIN=1` — DQV variants (default off)
- `SACV1=1` — V as side network (default off)
- `DUELING=1`, `DUELING_BETA` — Dueling SAC (default off, β=0.01)

Sweep driver: `run_dg_seeds.py` — configure CONFIGS list, runs 3 seeds each.
