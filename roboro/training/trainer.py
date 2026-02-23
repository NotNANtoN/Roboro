"""Minimal off-policy training loop — no framework, just a function."""

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch

from roboro.actors.base import BaseActor
from roboro.core.config import TrainCfg
from roboro.core.types import Batch
from roboro.data.replay_buffer import ReplayBuffer
from roboro.training.progress import ProgressTracker
from roboro.updates.base import BaseUpdate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Summary returned by ``train_off_policy``."""

    episode_rewards: list[float] = field(default_factory=list)
    eval_rewards: list[float] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)


def evaluate(
    env: gym.Env,
    actor: BaseActor,
    n_episodes: int = 5,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> float:
    """Run *n_episodes* greedy rollouts, return mean reward."""
    rewards = []
    for ep in range(n_episodes):
        # Seed each eval episode deterministically (if a base seed is given)
        ep_seed = seed + ep if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        total = 0.0
        done = False
        while not done:
            action, _info = actor.act(obs_t, deterministic=True)
            act_np = action.squeeze(0).cpu().numpy()
            # Handle discrete actions (scalar tensor -> int)
            act_np_env = int(act_np) if act_np.ndim == 0 else act_np
            obs, reward, terminated, truncated, _ = env.step(act_np_env)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    return sum(rewards) / len(rewards)


def train_off_policy(
    env: gym.Env,
    actor: BaseActor,
    update: BaseUpdate,
    buffer: ReplayBuffer,
    cfg: TrainCfg,
    device: torch.device | str = "cpu",
) -> TrainResult:
    """Generic off-policy training loop.

    1. Collect one transition with the actor.
    2. After ``warmup_steps``, sample a batch and call ``update.update()``.
    3. Periodically evaluate the actor greedily.

    Supports optional bfloat16 autocast via ``cfg.use_amp`` (CUDA / MPS only).

    Note: callers should call ``set_seed(cfg.seed)`` **before** creating models
    so that weight initialisation is also reproducible.
    """
    result = TrainResult()

    obs, _ = env.reset(seed=cfg.seed)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.0
    last_loss = float("nan")

    # AMP: only enable on accelerators, no-op on CPU
    device_type = str(device).split(":")[0]
    amp_enabled = cfg.use_amp and device_type != "cpu"

    with ProgressTracker(
        cfg.total_steps, show=cfg.show_progress, log_interval=cfg.log_interval
    ) as tracker:
        for step in range(1, cfg.total_steps + 1):
            # ── act ─────────────────────────────────────────────────────
            action, info = actor.act(obs_t)
            act_np = action.squeeze(0).cpu().numpy()
            act_np_env = int(act_np) if act_np.ndim == 0 else act_np

            next_obs, reward, terminated, truncated, _ = env.step(act_np_env)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=act_np,
                reward=float(reward),
                next_obs=next_obs,
                done=terminated,  # only true termination, not truncation
                **info,
            )

            obs = next_obs
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward += float(reward)

            if done:
                result.episode_rewards.append(episode_reward)
                tracker.log_episode(episode_reward)
                obs, _ = env.reset()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                episode_reward = 0.0

            # ── learn ───────────────────────────────────────────────────
            if (
                step >= cfg.warmup_steps
                and len(buffer) >= cfg.batch_size
                and step % cfg.train_freq == 0
            ):
                batch: Batch = buffer.sample(cfg.batch_size).to(device)
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp_enabled):
                    update_result = update.update(batch, step)
                last_loss = update_result.loss
                last_metrics = update_result.metrics
                result.metrics.append(
                    {"step": step, "loss": update_result.loss, **update_result.metrics}
                )

            # ── evaluate ────────────────────────────────────────────────
            if step % cfg.eval_interval == 0:
                eval_env = gym.make(env.spec.id) if env.spec else env  # type: ignore[union-attr]
                mean_reward = evaluate(
                    eval_env,
                    actor,
                    n_episodes=cfg.eval_episodes,
                    device=device,
                    seed=cfg.seed,
                )
                result.eval_rewards.append(mean_reward)
                tracker.log_eval(mean_reward)
                if env.spec:
                    eval_env.close()

            # ── progress ────────────────────────────────────────────────
            tracker.step(
                step,
                loss=last_loss,
                metrics=locals().get("last_metrics", None),
                actor=actor,
                buf_size=len(buffer),
            )

    return result
