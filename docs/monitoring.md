# Monitoring & Debugging RL Agents

Design decisions for what to track, why, and how.

---

## Tier 1 — Core (learning is broken without these)

| Signal | Why | Notes |
|---|---|---|
| **Episode reward curve** | Is the agent learning at all? | Rolling window of training episodes |
| **Evaluation reward** | True policy quality (no exploration noise) | Separate eval env, deterministic rollouts |
| **Critic loss** | Value function fitting? Diverging? | Per-component (critic, actor) |
| **Q-value mean / max** | Overestimation? Q-divergence? | Compare online vs target Q |
| **TD-error distribution** | Are Bellman targets reasonable? | Histogram of `|Q(s,a) - y|` |
| **Gradient norms** | Exploding / vanishing? | Per-component: encoder, critic, actor |

## Tier 2 — Diagnosis

| Signal | Why | Notes |
|---|---|---|
| **Eval video** | Sanity-check: is the policy doing something sensible? | `gymnasium.wrappers.RecordVideo` → wandb |
| **Action distribution** | Policy collapsed? Still exploring? | Mean, std, entropy of actions |
| **Exploration schedule** | ε / entropy coeff decaying correctly? | Already tracked for ε-greedy |
| **Replay buffer stats** | Stale data? Reward distribution shift? | Reward mean/std, done fraction, buffer age |
| **Target network divergence** | Online ↔ target drifting? | L2 distance between param vectors |
| **Learning rate** | Schedule correct? | Log current LR from optimizer |

## Tier 3 — Deep analysis

| Signal | Why | Notes |
|---|---|---|
| **Feature importance (SmoothGrad)** | Which input features drive Q / π? | See [SmoothGrad](#smoothgrad-feature-importance) below |
| **Eval on held-out dataset** | Offline RL: is the critic generalizing? | User-provided dataset → train/test split |
| **Q-value calibration** | Is Q(s,a) ≈ actual return? | Scatter plot: predicted Q vs Monte-Carlo return |
| **Representation diagnostics** | Dead neurons? Feature collapse? | Activation mean/var per layer, dead neuron ratio |
| **Weight norms per layer** | Init / regularization issues? | L2 norm of each layer's weights |
| **Policy KL divergence** | How fast is the policy changing? | KL(π_t ‖ π_{t−k}), useful for stability |

---

## SmoothGrad Feature Importance

The approach: perturb each input N times with Gaussian noise (σ ≈ 5% of
feature scale), compute the gradient of the output w.r.t. the input, and
average absolute gradients to get per-feature importance.

Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017.

```
for eval_obs in eval_batch:
    grads = []
    for _ in range(N):
        noisy = eval_obs + randn_like(eval_obs) * sigma
        noisy.requires_grad_(True)
        q = critic(noisy)
        q.backward()
        grads.append(noisy.grad.abs())
    importance = stack(grads).mean(0)   # (obs_dim,)
```

Log as a bar chart per feature dimension, or as a heatmap over time.

Applicable to both Q-functions (which features drive Q-values) and policies
(which features drive action selection). Compute during eval on a subsample.

---

## Offline RL Specifics

- **Dataset statistics dashboard**: reward distribution, episode lengths, state
  coverage visualization.
- **Train / test split**: user provides a transition dataset → auto-split;
  track critic loss on test set each eval interval.
- **OOD detection**: measure how far online Q-values deviate from the
  dataset's state distribution (relevant for CQL-style diagnostics).
- **Behavioral cloning baseline**: compare learned policy to the dataset's
  behavior policy as a lower bound.

---

## Architecture

A `Monitor` protocol with composable implementations:

```
class Monitor(Protocol):
    def on_step(self, step, batch, update_result, actor, critic): ...
    def on_episode_end(self, step, episode_reward): ...
    def on_eval(self, step, eval_env, actor, device): ...
    def close(self): ...
```

Concrete monitors: `WandbMonitor`, `VideoMonitor`, `SaliencyMonitor`,
`OfflineEvalMonitor`. The training loop calls hooks; each monitor handles
its own logging. This keeps the loop clean and concerns isolated.
