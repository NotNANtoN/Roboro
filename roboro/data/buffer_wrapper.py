import random

from roboro.data.replay_buffer import RLBuffer
from roboro.data.segment_tree import SumSegmentTree, MinSegmentTree
from roboro.utils import create_wrapper


def create_buffer(buffer_size, n_step, per, cer, gamma):
    # TODO: add PER, add simplified ERE (bias more recent transitions)
    # Create replay buffer
    update_freq = 0
    buffer_args = [buffer_size]
    buffer_kwargs = {'update_freq': update_freq}
    BufferClass = RLBuffer
    if per:
        BufferClass = create_wrapper(PER, BufferClass)
        buffer_kwargs.update({'beta_start': 0.4,
                              'alpha': 0.6})
    if n_step > 1:
        BufferClass = create_wrapper(NStep, BufferClass)
        buffer_kwargs.update({'n_step': n_step,
                              'gamma': gamma})
    if cer:
        BufferClass = create_wrapper(CER, BufferClass)
    buffer = BufferClass(*buffer_args, **buffer_kwargs)
    return buffer


class PER(RLBuffer):
    def __init__(self, buffer, max_priority=1.0, running_avg=0.0, beta_start=0.4, alpha=0.6):
        super().__init__(buffer)
        self.alpha = alpha
        self.max_priority = max_priority
        self.running_avg = running_avg
        self.beta = beta_start
        self.max_weight = None
        # Create Sum tres:
        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        weight = self._calc_weight(idx)
        extra_info = out[-1]
        extra_info["sample_weight"] = weight
        return out

    def add(self, *args, **kwargs):
        index = self.next_idx
        self._calculate_priority_of_last_add(index)
        super().add(*args, **kwargs)

    def sample_idx(self):
        mass = random.random() * self._it_sum.sum(0, len(self) - 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def update_priorities(self, idcs, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idcs) == len(priorities)
        for idx, priority in zip(idcs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            idx = int(idx)
            old_priority = self._it_sum[idx] * self.running_avg
            new_priority = old_priority + (priority ** self.alpha) * (1 - self.running_avg)
            self._it_sum[idx] = new_priority
            self._it_min[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

    def calc_and_save_max_weight(self):
        tree_min = self._it_min.min()
        tree_sum = self._it_sum.sum()
        p_min = tree_min / tree_sum
        self.max_weight = (p_min * len(self)) ** (-1 * self.beta)
        self.tree_sum = tree_sum

    def _calc_weight(self, index):
        p_sample = self._it_sum[index] / self.tree_sum
        weight = (p_sample * len(self)) ** (-1 * self.beta)
        weight /= self.max_weight
        return weight

    def _calculate_priority_of_last_add(self, idx):
        self._it_sum[idx] = self.max_priority ** self.alpha
        self._it_min[idx] = self.max_priority ** self.alpha

    def update(self, steps, extra_info):
        """ PER weight update, PER beta update"""
        pass
        # TODO: update beta from beta_start to beta_end in some way

        # TODO: also call self.calc_and_save_max_weight here, assuming it is done once per sampling


class NStep(RLBuffer):
    def __init__(self, *args, n_step=0, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_used = None

    def __str__(self):
        return f'NStep{self.n_step} <{super().__str__()}>'

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        extra_info = out[-1]
        assert self.n_step_used is not None, "get_reward() was not called, so number of steps per sample not known!"
        extra_info["n_step"] = self.n_step_used
        self.n_step_used = None
        return out

    def get_reward(self, idx):
        """ For n-step add rewards of discounted next n steps to current reward"""
        n_step_reward = 0
        count = 0
        for count, step_reward in enumerate(self.rewards[idx: idx + self.n_step]):
            n_step_reward += step_reward * self.gamma ** count
            if self.dones[idx + count]:
                break
        self.n_step_used = count + 1
        return n_step_reward

    def get_next_state(self, idx, state):
        count = 0
        done_slice = self.dones[idx: idx + self.n_step]
        for count, done in enumerate(done_slice):
            if done:
                break
        n_step_idx = idx + count
        return super().get_next_state(n_step_idx, state)


class CER(RLBuffer):
    def __init__(self, *args, **kwargs):
        """Returns the most recent experience tuple once per batch to bias new experiences slightly"""
        super().__init__(*args, **kwargs)
        self.new_batch = True

    def __str__(self):
        return f'CER <{super().__str__()}>'

    def __getitem__(self, idx):
        # overwrite index to be the most recently added index of the buffer at the start of a new batch
        if self.new_batch:
            idx = self.size() - 1
            self.new_batch = False
        return super().__getitem__(idx)

    def update(self, steps, extra_info):
        self.new_batch = True
        return super().update(steps, extra_info)
