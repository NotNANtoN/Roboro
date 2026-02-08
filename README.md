# Roboro

**roboro** — *to strengthen, to reinforce* (Latin)

A modular reinforcement learning library that decomposes algorithms into their true constituent techniques, making combinations transparent and experimentation fast.

## Design Principle

Algorithm names are aliases for combinations of orthogonal techniques. Roboro makes this explicit:

| Algorithm | Decomposition |
|-----------|--------------|
| **DQN** | Replay buffer + Q-network + ε-greedy + hard target net |
| **Rainbow** | DQN + PER + N-step + Dueling + NoisyNets + C51 + Double Q |
| **SAC** | Replay + clipped double Q + squashed Gaussian policy + max-entropy Bellman backup + auto α-tuning |
| **CrossQ** | SAC − target networks + BatchNorm in Q-networks |
| **TD3** | Replay + clipped double Q + deterministic policy + delayed actor update + target policy smoothing |
| **CQL** | SAC + conservative Q regularization (offline) |
| **IQL** | Expectile regression on V + advantage-weighted actor (offline) |
| **TD-MPC2** | Deterministic latent dynamics (joint embedding) + latent reward/value heads + TD(λ) in latent space + MPPI planning |
| **EfficientZero** | Latent dynamics (consistency loss) + latent reward/value + MCTS planning + self-supervised temporal consistency + learned value prefix |
| **Dreamer-v3** | Stochastic latent dynamics (RSSM) + observation reconstruction + latent reward/value + actor-critic in imagination |

## Architecture

```
roboro/
├── encoders/       Observation → latent features (MLP, CNN, VLM)
├── critics/        Value estimation (discrete Q, continuous Q, twin Q, distributional)
├── actors/         Action selection (ε-greedy, Gaussian, deterministic)
├── dynamics/       World models (deterministic, RSSM, hybrid)
├── planners/       Action optimization (MPPI, MCTS, CEM)
├── updates/        Learning rules (DQN, SAC, TD3, CQL, IQL, …)
├── exploration/    Intrinsic rewards (prediction error, ensemble disagreement, LP)
├── data/           Replay buffers (uniform, PER, N-step, HER) + offline datasets
├── envs/           Environment wrappers and utilities
├── algorithms/     Pre-wired recipes — each is a config composing the above
├── training/       Training loop, evaluation, logging (W&B)
└── core/           Shared types (Batch) and registry
```

### Key Design Decisions

1. **Critics are not policies.** A critic estimates value. An actor selects actions. They pair freely.
2. **Updates are composable functions**, not methods baked into critics.
3. **Target networks are a composable wrapper**, not an inheritance feature. CrossQ just doesn't use one.
4. **Algorithms are YAML configs** that wire components — not new classes.
5. **Offline RL is first-class.** The data module accepts environments *and* static datasets.
6. **Continuous actions are native.** Discrete is a special case, not the default.

## Quick Start

```bash
# Create environment and install
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Install pre-commit hooks
pre-commit install
```

## Development

```bash
# Lint & format
ruff check --fix .
ruff format .

# Type check
mypy roboro/

# Run tests with coverage
pytest --cov=roboro --cov-report=term-missing

# Run only fast tests (skip benchmarks)
pytest -m "not slow and not benchmark"
```

## Roadmap

- [x] Core abstractions (Batch, BaseEncoder, BaseCritic, BaseActor, BaseUpdate)
- [x] MLP encoder, discrete/continuous Q-critics, twin Q, target networks
- [x] Replay buffer with uniform sampling
- [x] Test infrastructure (pytest, ruff, mypy, pre-commit)
- [ ] **SAC** — first complete algorithm (continuous control)
- [ ] **DQN** — port from v1 onto new abstractions
- [ ] **Offline RL** — CQL, IQL, TD3+BC with D4RL datasets
- [ ] **TD-MPC2** — deterministic dynamics + MPPI planner
- [ ] **Dreamer-v3** — RSSM dynamics + imagination training
- [ ] **Exploration modules** — prediction error, ensemble disagreement
- [ ] **Performance benchmarks** — regression tests against known scores
- [ ] **Documentation site** — algorithm decomposition tables, tutorials

## Testing Philosophy

Inspired by Stable-Baselines3:

- **Unit tests**: every module produces correct output shapes, gradients flow
- **Smoke tests**: every algorithm config runs for 100 steps without crashing
- **Correctness tests**: SAC on Pendulum converges to ≥ −200 within 50k steps
- **Regression benchmarks**: tracked results on HalfCheetah, Ant, FetchReach

## License

GPL-3.0 — see [LICENSE](LICENSE).
