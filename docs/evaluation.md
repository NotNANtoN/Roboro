# Evaluation & Plotting

How to evaluate RL agents, what to plot, and best practices from the
literature.  This doc covers **post-training analysis** and **comparison
across runs**; see `monitoring.md` for in-loop signals.

---

## Core Plots — Learning Curves

Every experiment should produce at least these three views of the same
data.  Each answers a different question:

| X-axis | Y-axis | Question answered |
|---|---|---|
| **Env steps** | Eval reward | Sample efficiency — how many interactions? |
| **Episodes** | Eval reward | Convergence speed in terms of experience |
| **Wall time** | Eval reward | Practical speed — includes overhead, compilation, data loading |

Additionally:
- **Training reward** (with exploration) overlaid on eval reward to see
  the exploration–exploitation gap.
- **Rolling mean ± std** (e.g. window=10 evals) for noisy environments.
- **Per-seed curves + aggregate** when running multiple seeds (see below).

---

## Multi-Seed Reporting

A single seed proves nothing.  Follow [Agarwal et al., 2021 — *Deep RL
at the Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264):

| Metric | What it is | Why |
|---|---|---|
| **IQM** (Interquartile Mean) | Mean of the middle 50% of runs | Robust to outliers, recommended over plain mean |
| **Median** | 50th percentile | More stable than mean for heavy-tailed distributions |
| **Optimality gap** | How far below the target the best run is | Measures worst-case shortfall |
| **Probability of improvement** | P(algo A > algo B) over seeds | Direct comparison between methods |
| **Performance profiles** | CDF: fraction of runs achieving reward ≥ τ | Shows distribution, not just a point estimate |
| **Stratified bootstrap CI** | Confidence intervals via resampling | Proper uncertainty quantification (not ±1 std) |

**Minimum seeds**: 5 for smoke, 10+ for any publishable claim, 25+ for
statistical tests.

---

## Algorithm Comparison Plots

When comparing DQN vs Double DQN vs TD3 etc.:

| Plot | Description |
|---|---|
| **Overlay learning curves** | All algorithms on same axes, shaded CI |
| **Normalized score bar chart** | 0% = random, 100% = solved/expert.  Makes cross-env comparison fair |
| **Sample efficiency table** | Steps to reach reward X for each algo |
| **Wall-clock efficiency** | Same as above but in seconds — penalizes heavy computation |
| **Rank distribution** | How often each algo ranks 1st, 2nd, … across envs/seeds |

---

## Diagnostic Plots

Beyond reward, plot these to understand *why* an agent succeeds or fails:

| Plot | X-axis | Y-axis | Why |
|---|---|---|---|
| **Critic loss** | Steps | Loss | Divergence? Overfitting to replay? |
| **Q-value trajectory** | Steps | Mean / max Q | Overestimation (compare online vs target) |
| **TD-error histogram** | — | Count | Should be roughly centered near 0 |
| **Gradient norms** | Steps | Norm | Exploding / vanishing gradients |
| **Epsilon / entropy** | Steps | Value | Exploration schedule sanity check |
| **Action distribution** | Steps | Mean ± std | Policy collapse? Saturated actions? |
| **Replay buffer reward dist** | — | Histogram | Distribution shift over training |
| **Episode length** | Steps | Length | Is the agent surviving longer? (CartPole) or finishing faster? (goal-reaching) |

---

## Qualitative Evaluation

| Method | When | Notes |
|---|---|---|
| **Eval videos** | Every N evals | `gymnasium.wrappers.RecordVideo` → wandb media |
| **Side-by-side videos** | Comparison | Same initial state, different algorithms or checkpoints |
| **State visitation heatmap** | 2D envs | Where does the agent spend time? Coverage vs exploitation |
| **Attention / saliency maps** | Pixel inputs | SmoothGrad or GradCAM on encoder → what does the agent see? |

---

## Offline RL Specifics

| Plot | Description |
|---|---|
| **D4RL normalized score** | Standard comparison metric for offline RL |
| **Train vs test critic loss** | Split dataset; is the critic generalizing or memorizing? |
| **Q-value calibration scatter** | Predicted Q vs actual Monte-Carlo return — should be y=x |
| **OOD score distribution** | How far are online rollout states from the dataset? |
| **Dataset coverage vs performance** | Reward as a function of dataset quality / size |
| **Behavioral cloning baseline** | Lower bound — how good is just imitating the dataset? |

---

## Continuous Control Specifics

| Plot | Description |
|---|---|
| **Action magnitude histogram** | Are actions saturating at bounds? |
| **Joint torque profiles** | For robotics: per-joint action over an episode |
| **Smoothness metric** | Action jerk (d²a/dt²) — jerky policies are impractical |
| **Success rate curve** | For sparse-reward / goal-conditioned tasks |

---

## Implementation Plan

### Phase 1 — Core plotting utility

```
roboro/eval/plots.py

def plot_learning_curves(
    results: list[TrainResult],
    x: Literal["steps", "episodes", "wall_time"],
    title: str,
    rolling_window: int = 10,
    show_individual: bool = True,
) -> Figure:
    """Overlay multiple runs, shaded CI, IQM aggregate."""

def plot_comparison(
    algo_results: dict[str, list[TrainResult]],
    x: Literal["steps", "episodes", "wall_time"],
    normalize: tuple[float, float] | None = None,  # (random_score, expert_score)
) -> Figure:
    """Multi-algorithm comparison with bootstrap CIs."""
```

### Phase 2 — wandb integration

```
roboro/eval/wandb_logger.py

class WandbEvalLogger:
    def log_curves(self, results, step): ...
    def log_video(self, env, actor, step): ...
    def log_comparison_table(self, algo_results): ...
```

### Phase 3 — Statistical tools

```
roboro/eval/statistics.py

def iqm(scores: np.ndarray) -> float: ...
def stratified_bootstrap_ci(scores, n_bootstrap=10000, ci=0.95): ...
def probability_of_improvement(a_scores, b_scores): ...
def performance_profile(scores, thresholds): ...
```

Use the [`rliable`](https://github.com/google-research/rliable) library
(Agarwal et al.) for the statistical tools if available, with a pure-numpy
fallback.

---

## References

- Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., Bellemare, M.G.
  (2021). *Deep Reinforcement Learning at the Edge of the Statistical
  Precipice*. NeurIPS 2021.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.
  (2018). *Deep Reinforcement Learning that Matters*. AAAI 2018.
- Smilkov, D., Thorat, N., Kim, B., Viégas, F., Wattenberg, M. (2017).
  *SmoothGrad: removing noise by adding noise*. ICML Workshop.
