from typing import Any, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig, open_dict
from torch.optim.optimizer import Optimizer

from roboro.agent import Agent
from roboro.data import RLDataModule, create_buffer


class Learner(pl.LightningModule):
    """
    PyTorch Lightning Module that contains an Agent and trains it in an env

    Example:
        >>> import gymnasium as gym
        >>> from roboro.learner import Learner
        ...
        >>> learner = Learner(env="CartPole-v1")
    Train::
        trainer = Trainer()
        trainer.fit(learner, max_steps=10000)
    Note:
        Currently only supports CPU and single GPU training with `distributed_backend=dp`
    """

    def __init__(
        self,
        steps: int = 100000,
        train_env: str = None,
        train_ds: str = None,
        val_env: str = None,
        val_ds: str = None,
        test_env: str = None,
        test_ds: str = None,
        batch_size: int = 32,
        num_workers: int = 0,
        buffer_conf: DictConfig = None,
        warm_start_size: int = 1000,
        steps_per_batch: int = 1,
        sticky_actions: float = 0.0,
        frame_stack: int = 0,
        frameskip: int = 2,
        grayscale: int = 0,
        discretize_actions: bool = False,
        num_bins_per_dim: int = 5,
        render_mode: str = None,
        agent_conf: DictConfig = None,
        opt_conf: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Create replay buffer
        self.buffer, new_gamma = create_buffer(agent_conf.policy.gamma, **buffer_conf)
        with open_dict(agent_conf):
            agent_conf.policy.gamma = new_gamma
        print("Buffer: ", self.buffer)
        # Create envs and dataloaders
        self.datamodule = RLDataModule(
            self.buffer,
            train_env,
            train_ds,
            val_env,
            val_ds,
            test_env,
            test_ds,
            frame_stack=frame_stack,
            frameskip=frameskip,
            sticky_action_prob=sticky_actions,
            grayscale=grayscale,
            discretize_actions=discretize_actions,
            num_bins_per_dim=num_bins_per_dim,
            render_mode=render_mode,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_env, self.train_obs = self.datamodule.get_train_env()
        self.val_env, self.val_obs = self.datamodule.get_val_env()
        self.test_env, self.test_obs = self.datamodule.get_test_env()
        # init agent
        self.agent = Agent(
            self.train_env.observation_space,
            self.train_env.action_space,
            warm_start_steps=warm_start_size,
            **agent_conf,
        )
        self.agent.log = self.log
        self.agent.policy.log = self.log
        # print(self.agent)

        # init counters
        self.max_steps = steps
        self.total_steps = 0
        self.total_eps = 0
        self.epoch_steps = 0
        self.train_step_count = 0  # number of env steps to do this batch - gets incremented by steps_per_batch every batch
        # metrics to log
        self.mean_val_return = 0
        self.max_val_return = -np.inf
        self.n_evals = 0

        # tracking hyperparams:
        self.steps_per_epoch = min(max(self.max_steps / 100, 500), 20000)
        print("Steps per epoch: ", self.steps_per_epoch)
        self.steps_per_batch = steps_per_batch

        # hyperparams:
        self.warm_start = warm_start_size
        self.batch_size = batch_size
        self.frameskip = frameskip  # need to store here to keep track of real number of env interactions

        # optimizer hyperparams:
        self.opt_name = opt_conf.name
        self.lr = opt_conf.lr
        self.opt_eps = opt_conf.eps

    def forward(self, obs: Union[torch.Tensor, Tuple]) -> int:
        # If obs is a tuple, take the first element which should be the observation tensor
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(self.device, self.dtype).unsqueeze(0)
        action = self.agent(obs).item()
        return action

    def on_fit_start(self):
        """Fill the replay buffer with explorative experiences"""
        self.buffer.dtype = (
            self.dtype
        )  # give dtype to buffer for type compatibility in mixed precision
        if self.warm_start > 0:
            self.run(
                n_steps=self.warm_start, env=self.train_env, store=True, epsilon=1.0
            )
            self.train_obs = self.train_env.reset()[0]

    def on_train_start(self):
        """Do an evaluation round at the very start"""
        self.on_train_epoch_end()

    def on_train_epoch_start(self) -> None:
        """Reset counters"""
        self.buffer.should_stop = False
        self.epoch_steps = 0

    def on_train_batch_start(self, batch, batch_idx):
        """Take some steps in th the env and store them in the replay buffer"""
        if self.train_env is None:
            return
        self.train_step_count += self.steps_per_batch
        step_increase = 1 if self.frameskip <= 1 else self.frameskip
        while self.train_step_count > 1:
            self.train_step_count -= 1
            next_state, action, r, is_done = self.step_agent(
                self.train_obs, self.train_env, store=True
            )
            self.train_obs = next_state
            self.epoch_steps += step_increase
            self.total_steps += step_increase
            if is_done:
                self.train_obs = self.train_env.reset()[0]
                self.total_eps += 1
                if self.epoch_steps > self.steps_per_epoch:
                    #  force buffer to return None in next it
                    self.buffer.should_stop = True

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            batch_idx: idx of mini batch - not needed
        Returns:
            Training loss and log metrics
        """
        # calculates training loss
        loss, extra_info = self.agent.calc_loss(*batch)
        # update target nets, epsilon, etc:
        self.agent.update_self(self.total_steps)
        # update buffer
        self.buffer.update(self.total_steps / self.max_steps, extra_info)
        # log metrics
        self.log(
            "steps",
            self.total_steps,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "episodes",
            self.total_eps,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "epsilon",
            self.agent.epsilon,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Evaluate the agent for n episodes"""
        if self.val_env is None:
            return
        n = 5
        self.eval()
        val_reward_lists = self.run(self.val_env, n_eps=n)
        assert n == len(val_reward_lists)
        avg_return = sum(val_reward_lists) / n
        self.log(
            "val_ret",
            avg_return,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.n_evals += 1
        self.mean_val_return = (
            self.mean_val_return + (avg_return - self.mean_val_return) / self.n_evals
        )

        self.log(
            "mean",
            self.mean_val_return,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if avg_return > self.max_val_return:
            self.max_val_return = avg_return
        self.log(
            "max",
            self.max_val_return,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, *args, **kwargs):
        if batch is not None:
            pass

    def test_step(self, batch, *args, **kwargs):
        if batch is not None:
            pass

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Evaluate the agent for n episodes"""
        if self.test_env is None:
            return
        n = 5
        self.eval()
        test_reward_lists = self.run(self.test_env, n_eps=n)
        assert n == len(test_reward_lists)
        avg_return = sum(test_reward_lists) / n
        self.log(
            "test_return",
            avg_return,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def step_agent(self, obs, env, store=False):
        with torch.no_grad():
            action = self(obs)
        next_state, r, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated
        # add to buffer
        if store:
            self.buffer.add(state=obs, action=action, reward=r, done=is_done)
        return next_state, action, r, is_done

    def run(
        self,
        env,
        n_steps=0,
        n_eps: int = 0,
        epsilon: float = None,
        store=False,
        render=False,
    ) -> List[int]:
        """
        Carries out N episodes or N steps of the environment with the current agent
        Args:
            env: environment to use, either train environment or test environment
            n_eps: number of episodes to run
            n_steps: number of steps to run
            epsilon: epsilon value for DQN agent
            store: whether to store the experiences in the replay buffer
            render: whether to render the env
        """
        assert n_steps or n_eps
        agent_epsilon = self.agent.epsilon
        if epsilon is not None:
            self.agent.epsilon = epsilon
        total_rewards = []
        steps = 0
        eps = 0
        episode_state = env.reset()[0]

        while (not n_steps or steps < n_steps) and (not n_eps or eps < n_eps):
            is_done = False
            episode_reward = 0
            while not is_done:
                next_state, action, r, is_done = self.step_agent(
                    episode_state, env, store=store
                )
                episode_state = next_state
                episode_reward += r
                steps += 1
                if render:
                    env.render()
            episode_state = env.reset()[0]
            total_rewards.append(episode_reward)
            eps += 1
        self.agent.epsilon = agent_epsilon

        return total_rewards

    def configure_optimizers(self) -> Optimizer:
        if self.opt_name == "adam":
            optimizer = optim.Adam(
                self.agent.parameters(), lr=self.lr, eps=self.opt_eps
            )
        return optimizer

    def get_progress_bar_dict(self):
        """
        Don't show the version number in the progress bar
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        return items
