import random
from collections import defaultdict

import torch

from roboro.utils import apply_to_state


class RLBuffer(torch.utils.data.IterableDataset):
    def __init__(self, max_size, update_freq=0, gamma=0.99):
        super().__init__()
        self.max_size = max_size
        self.update_freq = update_freq
        self.gamma = gamma
        self.dtype = torch.float32  # can be overridden by trainer to be float16
        self.head = 0  # the index to which will be written next
        # Storage fields
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.extra_info = defaultdict(list)

        # Can bet set from the outside to determine the end of an epoch
        self.should_stop = False

    def __str__(self):
        return f"RLBuffer[{self.size()}, {self.max_size}]"

    def __iter__(self):
        count = 0
        while True:
            count += 1
            idx = self.sample_idx()
            yield self[idx]
            if self.should_stop or count == self.update_freq:
                return

    def __getitem__(self, idx):
        """ Return a single transition """
        if idx == self.decr_idx(self.head):
            idx = self.decr_idx(idx)
        # Stack states by calling .to():
        state = self.move(self.states[idx])
        next_state, done = self.get_next_state(idx, state)
        # Return extra info
        extra_info = {key: self.extra_info[key][idx] for key in self.extra_info}
        extra_info["idx"] = idx
        reward = self.get_reward(idx)
        return state, self.actions[idx], reward, done, next_state, extra_info

    def size(self):
        """Length method that does not override __len__, otherwise pytorch lightning would create a new epoch based on
        the buffer length"""
        return len(self.states) - 1  # subtract one because last added state can't be sampled (no next state yet)

    def sample_idx(self):
        idx = random.randint(0, self.size() - 1)  # subtract one because randint sampling includes the upper bound
        return idx

    def add(self, state, action, reward, done, store_episodes=False):
        # Mark episodic boundaries:
        # if self.dones[self.next_idx]:
        #    self.done_idcs.remove(self.next_idx)
        # if done:
        #    self.done_idcs.add(self.next_idx)

        # Store data:
        if len(self.states) == self.max_size:
            self.states[self.head] = state
            self.actions[self.head] = action
            self.rewards[self.head] = reward
            self.dones[self.head] = done
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        self.head = self.incr_idx(self.head)

    def get_reward(self, idx):
        """ Method that can be overridden by subclasses"""
        return float(self.rewards[idx])
        #return torch.tensor(self.rewards[idx], dtype=torch.float)

    def get_next_state(self, idx, state):
        """ Method that can be overridden by subclasses"""
        is_end = self.is_end(idx)
        if is_end:
            return torch.zeros_like(state), is_end
        else:
            next_state_idx = self.incr_idx(idx)
            return self.move(self.states[next_state_idx]), is_end

    def is_end(self, idx):
        return self.dones[idx] or idx == self.decr_idx(self.head)

    def update(self, train_frac, extra_info):
        """ PER weight update, PER beta update etc can happen here"""
        idcs = extra_info.pop("idx")
        for count, buffer_idx in enumerate(idcs):
            for key in extra_info:
                val = extra_info[key][count]
                self.add_extra_field(key, buffer_idx, val)

    def add_extra_field(self, key, idx, val):
        while len(self.extra_info[key]) < self.size() + 1:
            self.extra_info[key].append(torch.tensor(0))
        self.extra_info[key][idx] = val

    def move(self, obs):
        # Stack LazyFrames frames and convert to correct type (for half precision compatibility):
        return apply_to_state(lambda x: x.to("cpu", self.dtype), obs)

    def incr_idx(self, idx):
        idx = (idx + 1) % self.max_size
        return idx

    def decr_idx(self, idx):
        idx = (idx - 1)
        if idx < 0:
            idx = self.max_size - 1
        return idx

    def add_field(self, name, val):
        self.extra_info[name].append(val)
