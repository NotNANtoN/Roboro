from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import gym
from pytorch_lightning import seed_everything
from torch.optim.optimizer import Optimizer

from roboro.agent import Agent
from roboro.data import RLDataModule, RLBuffer


class Learner(pl.LightningModule):
    """
    PyTorch Lightning Module that contains an Agent and trains it in an train_env
    
    Example:
        >>> import gym
        >>> from roboro.learner import Learner
        ...
        >>> learner = Learner(train_env="CartPole-v1")
    Train::
        trainer = Trainer()
        trainer.fit(learner, max_steps=10000)
    Note:
        Currently only supports CPU and single GPU training with `distributed_backend=dp`
    """

    def __init__(self,
                 train_env: str = None,
                 train_ds: str = None,
                 val_env: str = None,
                 val_ds: str = None,
                 test_env: str = None,
                 test_ds: str = None,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 warm_start_size: int = 1000,
                 buffer_size=100000,
                 seed: int = 123,
                 steps_per_epoch: int = 100,
                 steps_per_batch: int = 1,
                 frame_stack: int = 0,
                 frameskip: int = 2,
                 grayscale: int = 0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # init train_env
        update_freq = 0
        self.buffer = RLBuffer(buffer_size, update_freq=update_freq)
        self.datamodule = RLDataModule(self.buffer,
                                       train_env, train_ds,
                                       val_env, val_ds,
                                       test_env, test_ds,
                                       frame_stack=frame_stack,
                                       frameskip=frameskip,
                                       grayscale=grayscale,
                                       norm_record_steps=warm_start_size)
        self.train_env, self.train_obs = self.datamodule.get_train_env()
        self.val_env, self.val_obs = self.datamodule.get_val_env()
        self.test_env, self.test_obs = self.datamodule.get_test_env()
        # init agent
        self.agent = Agent(self.train_env.observation_space, self.train_env.action_space)
        print(self.agent)

        # init counters
        self.total_steps = 0
        self.total_eps = 0
        self.epoch_steps = 0

        # tracking params:
        # TODO: calc steps per epoch properly
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_train = steps_per_batch

        # hyperparams:
        self.warm_start = warm_start_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.frameskip = frameskip

    def forward(self, obs: torch.Tensor) -> int:
        obs = obs.to(self.device, self.dtype)
        action = self.agent(obs).item()
        return action

    def on_fit_start(self):
        """Fill the replay buffer with explorative experiences"""
        if self.warm_start > 0:
            self.run(n_steps=self.warm_start, env=self.train_env, store=True, epsilon=1.0)
            self.train_obs = self.train_env.reset()

    def on_train_start(self):
        """ Do an evaluation round at the very start
        """
        self.training_epoch_end([])

    def on_train_epoch_start(self) -> None:
        """Reset counters"""
        self.buffer.should_stop = False
        self.total_steps += self.epoch_steps
        self.epoch_steps = 0

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """ Take some steps in th the train_env and store them in the replay buffer"""
        if self.train_env is None:
            return
        for _ in range(self.steps_per_train):
            next_state, action, r, is_done = self.step(self.train_obs, self.train_env, store=True)
            self.train_obs = next_state
            self.epoch_steps += 1 if self.frameskip <= 1 else self.frameskip
            if is_done:
                self.train_obs = self.train_env.reset()
                self.total_eps += 1
                if self.epoch_steps > self.steps_per_epoch:
                    #  force buffer to return None in next it
                    self.buffer.should_stop = True

    def training_step(self, batch, batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
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
        self.buffer.update(extra_info)
        # log metrics
        self.log('steps', self.total_steps, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('eps', self.total_eps, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """Evaluate the agent for n episodes"""
        if self.val_env is None:
            return
        n = 5
        self.eval()
        val_reward_lists = self.run(self.val_env, n_eps=n)
        assert n == len(val_reward_lists)
        avg_return = sum(val_reward_lists) / n
        self.log('val_return', avg_return, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, *args, **kwargs):
        if batch is not None:
            # TODO: do evaluation on random episodes of expert data
            pass

    def test_step(self, batch, *args, **kwargs):
        if batch is not None:
            # TODO: do evaluation on random episodes of expert data
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
        self.log('test_return', avg_return, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def step(self, obs, env, store=False):
        # send obs to correct device and data type. For frameskipping this also stacks LazyFrame objects into tensors
        obs = obs.to(self.device, self.dtype)
        action = self(obs)
        next_state, r, is_done, _ = env.step(action)
        # add to buffer
        if store:
            self.buffer.add(state=obs.squeeze(0), action=action, reward=r, done=is_done)
        return next_state, action, r, is_done

    def run(self, env, n_steps=0, n_eps: int = 0, epsilon: float = None, store=False, render=False) -> List[int]:
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
        episode_state = env.reset()

        while (not n_steps or steps < n_steps) and (not n_eps or eps < n_eps):
            is_done = False
            episode_reward = 0
            while not is_done:
                next_state, action, r, is_done = self.step(episode_state, env, store=store)
                episode_state = next_state
                episode_reward += r
                steps += 1
            episode_state = env.reset()
            total_rewards.append(episode_reward)
            eps += 1
            if render:
                env.render()
        self.agent.epsilon = agent_epsilon

        return total_rewards

    def configure_optimizers(self) -> Optimizer:
        optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)
        return optimizer

    def get_progress_bar_dict(self):
        """
        Don't show the version number in the progress bar
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        return items
