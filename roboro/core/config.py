"""Nested configuration dataclasses.

Shared sub-configs (``NetworkCfg``, ``TargetNetCfg``, …) are reusable across
algorithm families.  Each algorithm config (``DiscreteQCfg``,
``ContinuousActorCriticCfg``) composes them and adds its own knobs.

**Algorithm names are presets** — see ``roboro.presets``.

Compatible with Hydra / OmegaConf structured configs out of the box.

Example YAML (``configs/algorithm/dqn.yaml``)::

    network:
      hidden_dim: 128
      n_layers: 2
    target:
      mode: hard
      hard_update_period: 500
    buffer:
      capacity: 50_000
    train:
      total_steps: 50_000
    lr: 1e-3
    gamma: 0.99
    td_loss: huber
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── shared sub-configs ───────────────────────────────────────────────────────


@dataclass
class NetworkCfg:
    """MLP architecture shared by encoders, critics, actors."""

    hidden_dim: int = 256
    n_layers: int = 2
    activation: str = "relu"
    use_layer_norm: bool = False


@dataclass
class TargetNetCfg:
    """Target-network update strategy."""

    mode: str = "polyak"  # "polyak" | "hard"
    tau: float = 0.005
    hard_update_period: int = 1000


@dataclass
class BufferCfg:
    """Replay buffer sizing."""

    capacity: int = 100_000


@dataclass
class TrainCfg:
    """Training-loop control — shared by every off-policy algorithm."""

    total_steps: int = 50_000
    warmup_steps: int = 1_000
    batch_size: int = 128
    train_freq: int = 1  # gradient steps per N env steps (SB3 DQN uses 4)
    eval_interval: int = 5_000
    eval_episodes: int = 10
    log_interval: int = 500
    seed: int = 0
    show_progress: bool = True

    # Runtime
    device: str = "cpu"  # "cpu" | "cuda" | "mps"

    # Optional acceleration
    use_amp: bool = False  # bfloat16 autocast (CUDA / MPS)
    compile: bool = False  # torch.compile models before training


# ── algorithm configs ────────────────────────────────────────────────────────


@dataclass
class DiscreteQCfg:
    """Discrete Q-learning family.

    DQN, Double DQN, Dueling DQN, … are *presets* of this config.
    See ``roboro.presets``.
    """

    # Sub-configs (shared building blocks)
    network: NetworkCfg = field(default_factory=lambda: NetworkCfg(hidden_dim=128))
    target: TargetNetCfg = field(
        default_factory=lambda: TargetNetCfg(mode="hard", hard_update_period=500)
    )
    buffer: BufferCfg = field(default_factory=lambda: BufferCfg(capacity=50_000))
    train: TrainCfg = field(default_factory=TrainCfg)

    # Learning
    lr: float = 1e-3
    gamma: float = 0.99
    td_loss: str = "huber"  # "huber" (original DQN) | "mse"
    max_grad_norm: float = 10.0

    # Exploration schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000

    # Variants
    double_q: bool = False  # True → Double DQN (van Hasselt et al., 2016)


@dataclass
class ContinuousActorCriticCfg:
    """Continuous actor-critic family.

    DDPG, TD3, SAC are *presets* of this config.
    See ``roboro.presets``.
    """

    # Sub-configs
    actor_network: NetworkCfg = field(default_factory=lambda: NetworkCfg(hidden_dim=256))
    critic_network: NetworkCfg = field(default_factory=lambda: NetworkCfg(hidden_dim=256))
    target: TargetNetCfg = field(default_factory=lambda: TargetNetCfg(mode="polyak", tau=0.005))
    buffer: BufferCfg = field(default_factory=lambda: BufferCfg(capacity=100_000))
    train: TrainCfg = field(default_factory=lambda: TrainCfg(batch_size=256))

    # Learning
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    noise_std: float = 0.1

    # Variants (toggle for TD3, SAC, etc.)
    actor_type: str = "deterministic"  # "deterministic" (DDPG/TD3) | "squashed_gaussian" (SAC)
    twin_q: bool = False  # True → clipped double Q (TD3, SAC)
    actor_delay: int = 1  # 2 → delayed actor update (TD3)
    target_noise: float = 0.0  # > 0 → target policy smoothing (TD3)
    target_noise_clip: float = 0.5

    # SAC entropy (only used when actor_type == "squashed_gaussian")
    init_alpha: float = 1.0  # initial entropy coefficient
    learnable_alpha: bool = True  # auto-tune alpha toward target_entropy
    target_entropy: float | None = None  # default: -action_dim
    alpha_lr: float = 3e-4  # learning rate for log(alpha)
