import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import gym
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from roboro.agent import Agent
from roboro.env_wrappers import create_env


class Learner(pl.LightningModule):
    """
    PyTorch Lightning Module that contains an Agent and trains it in an train_env
    
    Example:
        >>> import gym
        >>> from roboro.learner import Learner
        ...
        >>> learner = Learner(train_env="CartPole-v0")
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
                 warm_start_size: int = 10000,
                 avg_reward_len: int = 100,
                 seed: int = 123,
                 steps_per_epoch: int = 1000,
                 frame_stack: int = 4,
                 frameskip: int = 4,
                 grayscale: int = 0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # init train_env
        assert train_env is not None or train_ds is not None, "Can't fit agent without training data!"
        self.train_env, self.train_dl = None, None
        if train_env is not None:
            self.train_env, self.train_obs = create_env(train_env, frameskip, frame_stack, grayscale)
        if train_ds is not None:
            self.train_dl = create_dl(train_ds)
            # TODO: make train/val/test split and use it
        # init val loader
        self.val_env, self.val_dl = self.train_env, self.train_dl
        if val_env is not None:
            self.val_env = create_env(val_env, frameskip, frame_stack, grayscale)
        if val_ds is not None:
            self.val_dl = create_dl(val_ds)
        elif self.train_dl is not None:
            self.val_dl = self.train_dl
        # init agent
        self.agent = Agent(self.train_env.sample(), self.train_env.action_space)

        # init counters
        self.total_steps = 0
        self.episode_counter = 0

        # tracking params:
        self.should_stop = False
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_train = 1

        # hyperparams:
        self.warm_start = warm_start_size
        self.batch_size = batch_size
        self.lr = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.agent(x)
        return output

    def training_step(self, batch):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
        Returns:
            Training loss and log metrics
        """
        if self.should_stop:
            return -1
        # calculates training loss
        loss, extra_info = self.agent.calc_loss(*batch)
        # update target nets, epsilon, etc:
        self.agent.update_self(self.total_steps)
        # update buffer
        self.agent.update_buffer(extra_info)

        self.log('steps', self.total_steps, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def val_step(self, *args, **kwargs):
        """Evaluate the agent for n episodes"""
        n = 5
        test_reward_lists = self.run(self.test_env, n_eps=n, epsilon=0)
        assert n == len(test_reward_lists)
        avg_return = sum(test_reward_lists) / n
        self.log('test_return', avg_return, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def step(self, obs, store=False):
        obs = obs.to(self.device, self.dtype)
        action = self(obs).squeeze(0)
        next_state, r, is_done, _ = self.train_env.step(action)
        # add to buffer
        if store:
            self.buffer.append(state=self.train_obs, action=action, reward=r, done=is_done, new_state=next_state)
        return next_state, action, r, is_done

    def on_batch_start(self):
        """ Determines how many steps to do in the train_env and runs them"""
        for _ in range(self.steps_per_train):
            next_state, action, r, is_done = self.step(self.train_obs, store=True)
            self.train_obs = next_state
            if is_done:
                self.train_obs = self.train_env.reset()
                self.episode_counter += 1
            self.total_steps += 1
            if self.total_steps % self.steps_per_epoch:
                self.should_stop = True

    def run(self, env, n_steps=0, n_eps: int = 0, epsilon: float = None, store=False) -> List[int]:
        """
        Carries out N episodes of the environment with the current agent
        Args:
            env: environment to use, either train environment or test environment
            n_eps: number of episodes to run
            n_steps: number of steps to run
            epsilon: epsilon value for DQN agent
            store: whether to store the experiences in the replay buffer
        """
        assert n_steps or n_eps

        eps = self.agent.epsilon
        if epsilon is not None:
            self.agent.epsilon = epsilon
        total_rewards = []
        steps = 0

        while (not n_steps or steps < n_steps) and (not n_eps or eps < n_eps):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                next_state, action, r, is_done = self.step(episode_state, store=store)

                episode_state = next_state
                episode_reward += r
                steps += 1
            total_rewards.append(episode_reward)
            eps += 1

        self.agent.epsilon = eps

        return total_rewards

    def on_train_start(self):
        if self.warm_start > 0:
            self.run(n_steps=self.warm_start, env=self.train_env, store=True, epsilon=1.0)

    def get_progress_bar_dict(self):
        """
        Don't show the version number in the progress bar
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
