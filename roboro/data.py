import random
from collections import deque, defaultdict

import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader

from roboro.env_wrappers import create_env


class RLBuffer(torch.utils.data.IterableDataset):
    def __init__(self, max_size, update_freq=0):
        self.update_freq = update_freq

        self.states = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.extra_info = defaultdict(deque)

        # Indexing fields:
        self.next_idx = 0
        self.curr_idx = 0
        self.looped_once = False
        self.should_stop = False
        
        # TODO: add n-step replay buffer by summing reward of future steps to current reward

    #def __len__(self):
    #    """ Return number of transitions stored so far """
    #    return len(self.states) - 1  # subtract one because last added state can't be sampled (no next state yet)

    def __getitem__(self, idx):
        """ Return a single transition """
        # Check if there is a next_state, if so stack frames:
        next_index = idx + 1
        is_end = self.dones[idx]
        # Stack states:
        state = self.states[idx].to("cpu")
        next_state = self.states[next_index].to("cpu") if not is_end else state
        # Return extra info
        extra_info = {key: self.extra_info[key][idx] for key in self.extra_info}
        extra_info["idx"] = idx
        return state, self.actions[idx], self.rewards[idx], is_end, next_state, extra_info

    def __iter__(self):
        count = 0
        while True:
            count += 1
            idx = self.sample_idx()
            yield self[idx]
            if self.should_stop or count == self.update_freq:
                return
                #raise StopIteration

    def sample_idx(self):
        num_entries = len(self.states) - 1  # subtract one because last added state can't be sampled (no next state yet)
        return random.randint(0, num_entries - 1)  # subtract one because randint samples including the upper bound

    def add(self, state, action, reward, done, store_episodes=False):
        # Mark episodic boundaries:
        # if self.dones[self.next_idx]:
        #    self.done_idcs.remove(self.next_idx)
        # if done:
        #    self.done_idcs.add(self.next_idx)

        # Store data:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def add_field(self, name, val):
        self.extra_info[name].append(val)

    def update(self, extra_info):
        pass


from torch.utils.data import random_split


class RLDataModule(pl.LightningDataModule):
    def __init__(self, buffer, train_env=None, train_ds=None, val_env=None, val_ds=None, test_env=None, test_ds=None,
                 batch_size=16,
                 **env_kwargs):
        super().__init__()
        assert train_env is not None or train_ds is not None, "Can't fit agent without training data!"
        self.buffer = buffer
        self.batch_size = batch_size

        self.train_env, self.train_dl = None, None
        if train_env is not None:
            self.train_env, self.train_obs = create_env(train_env, **env_kwargs)
        if train_ds is not None:
            #self.dev_dataset = create_dl(train_ds)
            #self.train_data, self.val_data = random_split(dev_data, [55000, 5000])
            # TODO: make train/val/test split and use it
            # TODO: somehow extract obs wrapper from env and use it in dataloader (what if there is no env?)
            pass
        # init val loader
        self.val_env, self.val_dl = self.train_env, self.train_dl
        if val_env is not None:
            self.val_env = create_env(val_env, **env_kwargs)
        self.val_obs = self.val_env.reset()
        if val_ds is not None:
            pass
            #self.val_dl = create_dl(val_ds)
        # init test_env
        self.test_env, self.test_dl = self.val_env, self.val_dl
        if test_env is not None:
            self.test_env = create_env(test_env, **env_kwargs)
        self.test_obs = self.test_env.reset()

        #if val_ds is not None:
        #    self.val_dl = create_dl(val_ds)

    def collate(self, batch):
        print(batch)
        print(len(batch))
        quit()

    def _dataloader(self, ds) -> DataLoader:
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=0)#, collate_fn=self.collate)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        # TODO: combine replay buffer dataloader with expert data dataloader
        return self._dataloader(self.buffer)

    def val_dataloader(self) -> DataLoader:
        return self.val_dl
        #return self.train_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self.test_dl

    def get_train_env(self):
        return self.train_env, self.train_obs

    def get_val_env(self):
        return self.val_env, self.val_obs

    def get_test_env(self):
        return self.test_env, self.test_obs
        #return DataLoader(self.test, batch_size=32)
