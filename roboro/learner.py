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
                 eps_start: float = 1.0,
                 eps_end: float = 0.02,
                 eps_last_frame: int = 150000,
                 sync_rate: int = 1000,
                 gamma: float = 0.99,
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 replay_size: int = 100000,
                 warm_start_size: int = 10000,
                 avg_reward_len: int = 100,
                 min_episode_reward: int = -21,
                 seed: int = 123,
                 batches_per_epoch: int = 1000,
                 n_steps: int = 1,
                 **kwargs):
        super().__init__()
        self.agent = agent
        self.env = env if not (type(env) == str) else gym.make(env)

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

        # calculates training loss
        loss = self.agent.calc_loss(*batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log('steps', self.global_step, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, *args, **kwargs):
        """Evaluate the agent for 10 episodes"""
        test_reward = self.run(self.test_env, n_eps=5, epsilon=0)
        avg_reward = sum(test_reward) / len(test_reward)
        self.log('test_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def train_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while self.total_steps % self.batches_per_epoch != 0:
            self.total_steps += 1
            # Act
            action = self.agent(self.state)
            # Env step
            next_state, r, is_done, _ = self.env.step(action[0])
            episode_reward += r
            episode_steps += 1
            # Buffer update
            self.agent.update_epsilon(self.global_step)
            self.buffer.append(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)
            self.state = next_state

            if is_done:
                # self.done_episodes += 1
                # self.total_rewards.append(episode_reward)
                # self.total_episode_steps.append(episode_steps)
                # self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.env.reset()
                # episode_steps = 0
                # episode_reward = 0

            # Sample train batch:
            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)
            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

    def run(self, env, n_steps=0, n_eps: int = 0, epsilon: float = 1.0, store=False) -> List[int]:
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
        total_rewards = []
        steps = 0

        while (not n_steps or steps < n_steps) and (not n_eps or eps < n_eps):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
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

    def populate(self, warm_start):
        if warm_start > 0:
            self.run(n_steps=warm_start, env=self.env, store=True, epsilon=1.0)

    def get_progress_bar_dict(self):
        """
        Don't show the version number in the progress bar
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
