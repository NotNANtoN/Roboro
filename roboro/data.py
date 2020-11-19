from collections import namedtuple

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from roboro.env_wrappers import create_env


def train_batch(self):
    """
    Contains the logic for generating a new batch of data to be passed to the DataLoader
    Returns:
        yields a Experience tuple containing the state, action, reward, done and next_state.
    """
    episode_reward = 0
    episode_steps = 0

    while self.total_steps % self.batches_per_epoch != 0:

        # Sample train batch:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)
        for idx, _ in enumerate(dones):
            yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]


class ExperienceSourceDataset(torch.utils.data.IterableDataset):
    """
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch):
        self.generate_batch = generate_batch

    def __iter__(self):
        iterator = self.generate_batch()
        return iterator


class RLDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_size, obs_sample, act_shape):
        self.states = torch.empty([max_size] + obs_sample.shape, dtype=obs_sample.dtype)
        self.rewards = torch.empty(max_size)
        self.actions = torch.empty([max_size] + act_shape)
        self.dones = torch.empty(max_size, dtype=torch.bool)

        # Indexing fields:
        self.next_idx = 0
        self.curr_idx = 0
        self.looped_once = False

    def __len__(self):
        """ Return number of transitions stored so far """
        if self.looped_once:
            return self.max_size
        else:
            return self.next_idx

    def __getitem__(self, index):
        """ Return a single transition """
        # Check if the last state is being attempted to sampled - it has no next state yet:
        if index == self.curr_idx:
            index = self.decrement_idx(index)
        elif index >= len(self):
            raise ValueError("Error: index " + str(index) + " is too large for buffer of size " + str(len(self)))
        # Check if there is a next_state, if so stack frames:
        next_index = self.increment_idx(index)
        is_end = self.is_episode_boundary(index)
        # Stack states:
        state = self.stack_last_frames_idx(index)
        next_state = self.stack_last_frames_idx(next_index) if not is_end else None
        return [state, self.actions[index].squeeze(), self.rewards[index].squeeze(), next_state, torch.tensor(index)]

    def __iter__(self):
        count = 0
        while True:
            count += 1
            idx = self.sample_idx()
            yield self[idx]
            if count == self.update_freq:
                return
                #raise StopIteration

    def add(self, state, action, reward, done, store_episodes=False):
        # Mark episodic boundaries:
        # if self.dones[self.next_idx]:
        #    self.done_idcs.remove(self.next_idx)
        # if done:
        #    self.done_idcs.add(self.next_idx)

        # Store data:
        if self.use_list and not self.looped_once:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states[self.next_idx] = state
            self.actions[self.next_idx] = action
            self.rewards[self.next_idx] = reward
            self.dones[self.next_idx] = done

        # Take care of idcs:
        self.curr_idx = self.next_idx
        self.next_idx = self.increment_idx(self.next_idx)

    def increment_idx(self, index):
        """ Loop the idx from front to end. Skip expert data, as we want to keep that forever"""
        index += 1
        if index == self.max_size:
            index = 0 + self.size_expert_data
            self.looped_once = True
        return index

    def decrement_idx(self, index):
        index -= 1
        if index < 0:
            index = len(self) - 1
        return index



class RLDataModule(pl.LightningDataModule):
    def __init__(self, replay_buffer, val_env=None, test_env=None, batch_size=16):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        return RLDataLoader(self.replay_buffer, self.batch_size)


        #self.dataset = ExperienceSourceDataset(self.train_batch)
        #return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size(). Defined in self.setup()
        self.dims = None

    def prepare_data(self):
        # download and preprocess once
        #MNIST(self.data_dir, train=True, download=True)
        #MNIST(self.data_dir, train=False, download=True)
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders. Done on every GPU
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)