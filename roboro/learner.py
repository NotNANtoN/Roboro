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
#from pl_bolts.datamodules.experience_source import ExperienceSourceDataset, Experience
#from pl_bolts.models.rl.common.memory import MultiStepBuffer


class Learner(pl.LightningModule):
    """
    PyTorch Lightning Module that contains an Agent and trains it in an env
    
    Example:
        >>> import gym
        >>> from roboro.agent import Agent
        >>> from roboro.learner import Learner
        ...
        >>> env = gym.make("CartPole-v0")
        >>> agent = Agent(env.observation_space, env.action_space)
        >>> learner = Learner(agent, env)
    Train::
        trainer = Trainer()
        trainer.fit(learner)
    Note:
        Currently only supports CPU and single GPU training with `distributed_backend=dp`
    """

    def __init__(self,
                 agent: Agent,
                 env,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 warm_start_size: int = 10000,
                 avg_reward_len: int = 100,
                 seed: int = 123,
                 steps_per_epoch: int = 1000,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.agent = agent
        self.env = env if not (type(env) == str) else gym.make(env)
        self.state = self.env.reset()

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

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        self.log('steps', self.total_steps, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def val_step(self, *args, **kwargs):
        """Evaluate the agent for 10 episodes"""
        test_reward = self.run(self.test_env, n_eps=5, epsilon=0)
        avg_reward = sum(test_reward) / len(test_reward)
        self.log('test_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def on_batch_start(self):
        """ Determines how many steps to do in the env"""
        for _ in range(self.steps_per_train):
            action = self.agent(self.state).squeeze(0)
            next_state, r, is_done, _ = self.env.step(action)
            # add to buffer
            self.buffer.append(state=self.state, action=action, reward=r, done=is_done, new_state=next_state)
            self.state = next_state
            if is_done:
                self.state = self.env.reset()
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

                action = self.agent(episode_state, self.device).squeeze(0)
                next_state, reward, done, _ = self.env.step(action)
                episode_state = next_state
                episode_reward += reward
                if store:
                    self.buffer.append(episode_state, action, reward, done, next_state)
                steps += 1
            total_rewards.append(episode_reward)

        self.agent.epsilon = eps

        return total_rewards

    def on_train_start(self):
        if self.warm_start > 0:
            self.run(n_steps=self.warm_start, env=self.env, store=True, epsilon=1.0)

    def get_progress_bar_dict(self):
        """
        Don't show the version number in the progress bar
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
