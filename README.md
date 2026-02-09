# Roboro

**roboro** — *to strengthen, to reinforce* (Latin)

A modular reinforcement learning library that decomposes algorithms into their true constituent techniques, making combinations transparent and experimentation fast.

## Design Principle

Algorithm names are aliases for combinations of orthogonal techniques. Roboro makes this explicit:

### Discrete Q-learning family

| Technique | DQN | Double DQN | Rainbow |
|---|---|---|---|
| Replay buffer | uniform | uniform | PER |
| Q-network | single | single | dueling + distributional (C51) |
| Exploration | ε-greedy | ε-greedy | NoisyNets |
| Target network | hard copy | hard copy | hard copy |
| Action selection | argmax Q | argmax Q (online selects, target evaluates) | argmax Q |
| N-step returns | — | — | ✓ |

### Continuous actor-critic family

| Technique | DDPG | TD3 | SAC | CrossQ |
|---|---|---|---|---|
| Actor | deterministic | deterministic | squashed Gaussian | squashed Gaussian |
| Q-critics | single | **twin** (clipped double) | twin | twin |
| Exploration | Gaussian noise | Gaussian noise | **entropy-driven** | entropy-driven |
| Target networks | Polyak (actor + critic) | Polyak (actor + critic) | Polyak (critic only) | **none** (BatchNorm) |
| Delayed actor update | — | **every 2 steps** | — | — |
| Target policy smoothing | — | **clipped noise** | — | — |
| Entropy tuning | — | — | **auto α** | auto α |

### Offline RL family (planned)

| Technique | CQL | IQL | TD3+BC |
|---|---|---|---|
| Base | SAC | separate V + advantage | TD3 |
| Offline constraint | conservative Q penalty | expectile regression | BC regularization |

### Model-based family (planned)

| Algorithm | Decomposition |
|-----------|--------------|
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
└── core/           Shared types (Batch), seeding, device utilities
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

## Status

**Implemented:** DQN, Double DQN, DDPG, TD3, SAC — with unit tests, smoke tests, initial benchmarks, and deterministic seeding.

**Next up:** monitoring/evaluation infrastructure, full performance benchmarks, offline RL.

See **[ROADMAP.md](ROADMAP.md)** for the full task list and priorities.

## Testing Philosophy

Inspired by Stable-Baselines3:

- **Unit tests**: every module produces correct output shapes, gradients flow
- **Smoke tests**: every algorithm config runs for ~5k steps without crashing
- **Performance benchmarks**: DQN on CartPole, SAC/DDPG/TD3 on Pendulum, etc.
- **Reproducibility tests**: identical seeds produce identical results

## License

GPL-3.0 — see [LICENSE](LICENSE).
