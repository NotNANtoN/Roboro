"""Minimal off-policy training loop — no framework, just a function."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch

from roboro.actors.base import BaseActor
from roboro.core.types import Batch
from roboro.data.replay_buffer import ReplayBuffer
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
) -> float:
    """Run *n_episodes* greedy rollouts, return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        total = 0.0
        done = False
        while not done:
            action = actor.act(obs_t, deterministic=True)
            act_np = action.squeeze(0).cpu().numpy()
            # Handle discrete actions (scalar tensor → int)
            if act_np.ndim == 0:
                act_np = int(act_np)
            obs, reward, terminated, truncated, _ = env.step(act_np)
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
    total_steps: int,
    batch_size: int = 256,
    warmup_steps: int = 1000,
    eval_interval: int = 2000,
    eval_episodes: int = 5,
    device: torch.device | str = "cpu",
    log_interval: int = 500,
) -> TrainResult:
    """Generic off-policy training loop.

    1. Collect one transition with the actor.
    2. After ``warmup_steps``, sample a batch and call ``update.update()``.
    3. Periodically evaluate the actor greedily.

    Works for both DQN (discrete) and DDPG (continuous).
    """
    result = TrainResult()

    obs, _ = env.reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.0

    for step in range(1, total_steps + 1):
        # ── act ─────────────────────────────────────────────────────────
        action = actor.act(obs_t)
        act_np = action.squeeze(0).cpu().numpy()
        # Handle discrete actions
        if act_np.ndim == 0:
            act_np = int(act_np)

        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        done = terminated or truncated

        # Store as flat tensors
        buffer.add(
            obs=torch.as_tensor(obs, dtype=torch.float32),
            action=action.squeeze(0).cpu().float(),
            reward=float(reward),
            next_obs=torch.as_tensor(next_obs, dtype=torch.float32),
            done=terminated,  # only true termination, not truncation
        )

        obs = next_obs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward += float(reward)

        if done:
            result.episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0.0

        # ── learn ───────────────────────────────────────────────────────
        if step >= warmup_steps and len(buffer) >= batch_size:
            batch: Batch = buffer.sample(batch_size).to(device)
            update_result = update.update(batch, step)
            result.metrics.append({"step": step, **update_result.metrics})

            if step % log_interval == 0:
                logger.info(
                    "step=%d  loss=%.4f  %s",
                    step,
                    update_result.loss,
                    {k: f"{v:.3f}" for k, v in update_result.metrics.items()},
                )

        # ── evaluate ────────────────────────────────────────────────────
        if step % eval_interval == 0:
            eval_env = gym.make(env.spec.id) if env.spec else env  # type: ignore[union-attr]
            mean_reward = evaluate(eval_env, actor, n_episodes=eval_episodes, device=device)
            result.eval_rewards.append(mean_reward)
            if env.spec:
                eval_env.close()
            logger.info("step=%d  eval_reward=%.2f", step, mean_reward)

    return result
