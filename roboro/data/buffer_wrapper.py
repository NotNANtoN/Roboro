import random

import torch

from roboro.data.replay_buffer import RLBuffer
from roboro.data.segment_tree import MinSegmentTree, SumSegmentTree
from roboro.utils import create_wrapper


def create_buffer(gamma, buffer_size=100000, n_step=0, per=0, cer=0, **buffer_kwargs):
    # Create replay buffer
    buffer_args = [buffer_size]
    # buffer_kwargs.update{'update_freq': update_freq}
    BufferClass = RLBuffer
    if per:
        BufferClass = create_wrapper(PER, BufferClass)
        # buffer_kwargs.update({'beta_start': 0.4,
        #                      'alpha': 0.6})
    if n_step > 1:
        BufferClass = create_wrapper(NStep, BufferClass)
        buffer_kwargs.update({"n_step": n_step, "gamma": gamma})
        gamma = gamma**n_step
    if cer:
        BufferClass = create_wrapper(CER, BufferClass)
    buffer = BufferClass(*buffer_args, **buffer_kwargs)
    return buffer, gamma


class PER(RLBuffer):
    def __init__(
        self,
        *args,
        max_priority=1.0,
        running_avg=0.0,
        beta_start=0.4,
        alpha=0.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.max_priority = max_priority
        self.running_avg = running_avg
        self.beta_start = beta_start
        self.beta = beta_start
        self.max_weight = None
        # Create Sum tres:
        it_capacity = 1
        while it_capacity < self.max_size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        # caches tree sum during sampling of a batch
        self.tree_sum = None

    def __str__(self):
        return f"PER{self.alpha}_{self.beta_start}<{super().__str__()}>"

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        weight = self._calc_weight(idx)
        extra_info = out[-1]
        extra_info["sample_weight"] = weight
        return out

    def add(self, *args, **kwargs):
        self._set_priority_of_new_exp(self.head)
        super().add(*args, **kwargs)

    def sample_idx(self):
        if self.tree_sum is None:
            self.calc_and_save_max_weight()
        mass = random.random() * self.tree_sum  # self._it_sum.sum(0, self.size() + 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def update_priorities(self, idcs, priorities):
        """
        Update priorities of sampled transitions.
        Sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idcs: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idcs`.
        """
        if torch.is_tensor(idcs):
            idcs = idcs.int().tolist()
        if torch.is_tensor(priorities):
            priorities = priorities.tolist()
        assert len(idcs) == len(priorities)
        for idx, priority in zip(idcs, priorities):
            if priority == 0:
                priority += 0.001
            assert priority > 0, f"priority: {priority}"
            assert 0 <= idx <= self.size(), f"idx: {idx}"
            old_priority = 0
            if self.running_avg:
                old_priority = self._it_sum[idx] * self.running_avg
            new_priority = old_priority + (priority**self.alpha) * (
                1 - self.running_avg
            )
            self._it_sum[idx] = new_priority
            self._it_min[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

    def calc_and_save_max_weight(self):
        """Needs to be called before sampling a batch<"""
        tree_min = self._it_min.min()
        tree_sum = self._it_sum.sum()
        p_min = tree_min / tree_sum
        self.max_weight = (p_min * (self.size() + 1)) ** (-self.beta)
        self.tree_sum = tree_sum

    def _calc_weight(self, index):
        p_sample = self._it_sum[index] / self.tree_sum
        weight = (p_sample * (self.size() + 1)) ** (-self.beta)
        weight /= self.max_weight
        return weight

    def _set_priority_of_new_exp(self, idx):
        self._it_sum[idx] = self.max_priority**self.alpha
        self._it_min[idx] = self.max_priority**self.alpha

    def update(self, train_frac, extra_info):
        """PER weight update, PER beta update"""
        idcs = extra_info["idx"]
        tde = extra_info["tde"]
        self.update_priorities(idcs, tde)
        self.calc_and_save_max_weight()
        self.beta = self.beta_start + (1 - self.beta_start) * train_frac
        super().update(train_frac, extra_info)


class NStep(RLBuffer):
    def __init__(self, *args, n_step=0, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_step = n_step
        self.gamma = gamma

    def __str__(self):
        return f"NStep{self.n_step}<{super().__str__()}>"

    def get_reward(self, idx):
        """For n-step add rewards of discounted next n steps to current reward"""
        n_step_reward = 0
        step_idx = idx
        for count in range(self.n_step):
            step_reward = super().get_reward(step_idx)
            step_idx = self.incr_idx(step_idx)
            n_step_reward += step_reward * self.gamma**count
            if self.is_end(step_idx):
                break
        return n_step_reward

    def get_next_state(self, idx, state):
        n_step_idx = idx
        for _ in range(self.n_step):
            if self.is_end(n_step_idx):
                break
            n_step_idx = self.incr_idx(n_step_idx)
        return super().get_next_state(n_step_idx, state)


class CER(RLBuffer):
    def __init__(self, *args, **kwargs):
        """Returns the most recent experience tuple once per batch to bias new experiences slightly"""
        super().__init__(*args, **kwargs)
        self.new_batch = True

    def __str__(self):
        return f"CER<{super().__str__()}>"

    def __getitem__(self, idx):
        # overwrite index to be the most recently added index of the buffer at the start of a new batch
        if self.new_batch:
            idx = self.decr_idx(self.decr_idx(self.head))
            self.new_batch = False
        return super().__getitem__(idx)

    def update(self, steps, extra_info):
        self.new_batch = True
        return super().update(steps, extra_info)
