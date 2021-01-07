import random
import itertools
from collections import deque, defaultdict

import torch

from roboro.utils import apply_to_state


class SliceableDeque(deque):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(itertools.islice(self, index.start,
                                               index.stop, index.step))
        return deque.__getitem__(self, index)


class RLBuffer(torch.utils.data.IterableDataset):
    def __init__(self, max_size, update_freq=0, n_step=0, gamma=0.99):
        super().__init__()
        self.max_size = max_size
        self.update_freq = update_freq
        self.n_step = n_step
        self.gamma = gamma
        self.dtype = torch.float32  # can be overridden by trainer to be float16
        # Storage fields
        self.states = SliceableDeque(maxlen=max_size)
        self.rewards = SliceableDeque(maxlen=max_size)
        self.actions = SliceableDeque(maxlen=max_size)
        self.dones = SliceableDeque(maxlen=max_size)
        self.extra_info = defaultdict(SliceableDeque)

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
                #raise StopIteration

    def __getitem__(self, idx):
        """ Return a single transition """
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
        return random.randint(0, self.size() - 1)  # subtract one because randint sampling includes the upper bound

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

    def get_reward(self, idx):
        """ Method that can be overridden by subclasses"""
        return torch.tensor(self.rewards[idx], dtype=torch.float)

    def get_next_state(self, idx, state):
        """ Method that can be overridden by subclasses"""
        is_end = self.is_end(idx)
        if is_end:
            return torch.zeros_like(state), is_end
        else:
            return self.move(self.states[idx + 1]), is_end

    def is_end(self, idx):
        return self.dones[idx] or idx == self.size()

    def update(self, train_frac, extra_info):
        """ PER weight update, PER beta update etc can happen here"""
        pass

    def move(self, obs):
        # Stack LazyFrames frames and convert to correct type (for half precision compatibility):
        return apply_to_state(lambda x: x.to("cpu", self.dtype), obs)

    def add_field(self, name, val):
        self.extra_info[name].append(val)
