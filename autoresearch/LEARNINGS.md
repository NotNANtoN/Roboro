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
