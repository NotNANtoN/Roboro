import random

import numpy as np
import torch

from roboro.data.replay_buffer import RLBuffer
from roboro.data.segment_tree import MinSegmentTree, SumSegmentTree
from roboro.utils import create_wrapper


def create_buffer(
    gamma, buffer_size=100000, n_step=0, per=0, cer=0, her=0, **buffer_kwargs
):
    # Create replay buffer
    buffer_args = [buffer_size]
    # buffer_kwargs.update{'update_freq': update_freq}
    BufferClass = RLBuffer  # noqa: N806
    if per:
        BufferClass = create_wrapper(PER, BufferClass)  # noqa: N806
        # buffer_kwargs.update({'beta_start': 0.4,
        #                      'alpha': 0.6})
    if n_step > 1:
        BufferClass = create_wrapper(NStep, BufferClass)  # noqa: N806
        buffer_kwargs.update({"n_step": n_step, "gamma": gamma})
        gamma = gamma**n_step
    if cer:
        BufferClass = create_wrapper(CER, BufferClass)  # noqa: N806
    if her:
        BufferClass = create_wrapper(HER, BufferClass)  # noqa: N806
        # Default HER parameters can be overridden in buffer_kwargs
    buffer = BufferClass(*buffer_args, **buffer_kwargs)
    return buffer, gamma


class HER(RLBuffer):
    """Hindsight Experience Replay (HER) buffer wrapper.

    Retrospectively substitutes goals to turn failed experiences into successful ones.
    Assumes observations are flattened goal-conditioned: [obs, achieved_goal, desired_goal].
    """

    def __init__(
        self,
        *args,
        her_ratio=0.8,  # Fraction of batch to replace with HER experiences
        her_strategy="future",  # "future", "final", "episode", "random"
        goal_dim=3,  # Dimension of goal space (e.g., 3D position)
        obs_dim=10,  # Dimension of robot observation (without goals)
        reward_fn=None,  # Custom reward function, defaults to goal distance
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.her_ratio = her_ratio
        self.her_strategy = her_strategy
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.reward_fn = reward_fn or self._default_reward_fn

        # Storage for episode tracking
        self.current_episode = []  # Store transitions for current episode
        self.episodes = []  # Store completed episodes for "random" strategy
        self.max_episodes = 1000  # Maximum episodes to keep for "random" strategy

        print(
            f"HER enabled: ratio={her_ratio}, strategy={her_strategy}, goal_dim={goal_dim}"
        )

    def __str__(self):
        return f"HER{self.her_ratio}_{self.her_strategy}<{super().__str__()}>"

    def add(self, state, action, reward, done, **kwargs):
        # Ensure state is stored with correct dtype for MPS compatibility
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state.astype(np.float32))
        elif isinstance(state, torch.Tensor) and state.dtype != torch.float32:
            state = state.float()

        # Extract goals from flattened observation
        achieved_goal, desired_goal = self._extract_goals(state)

        # Store transition in current episode
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        self.current_episode.append(transition)

        # Add original transition to buffer
        super().add(state, action, reward, done, **kwargs)

        # If episode ended, process HER and store episode
        if done:
            self._process_episode_her()
            # Store episode for "random" strategy
            if self.her_strategy == "random":
                self.episodes.append(self.current_episode.copy())
                if len(self.episodes) > self.max_episodes:
                    self.episodes.pop(0)
            self.current_episode = []

    def __getitem__(self, idx):
        # Get original transition
        state, action, reward_tensor, done, next_state, extra_info = (
            super().__getitem__(idx)
        )

        # Apply HER with probability her_ratio
        # if random.random() < self.her_ratio and len(self.current_episode) > 1:
        #     # --- OPTION C NOTE ---
        #     # To correctly implement on-the-fly HER ("future", "final", "episode" strategies)
        #     # for an arbitrary transition `idx`, the buffer would need to be able to
        #     # retrieve or reconstruct the full episode to which `idx` belongs.
        #     # This would allow `_sample_her_goal` (below) to sample goals
        #     # from the *actual context* of the original experience.
        #     # This typically involves storing episodes separately or having a mechanism
        #     # to trace back episode boundaries from `idx`.
        #     #
        #     # The current implementation of _sample_her_goal (for non-"random" strategies)
        #     # incorrectly uses `self.current_episode` (the episode currently being recorded)
        #     # rather than the episode of the transition `idx`.
        #     #
        #     # For now, this on-the-fly relabeling is DISABLED.
        #     # The primary HER mechanism is via `_process_episode_her` which correctly
        #     # uses future states from the *same* episode upon its completion.
        #
        #     # Create HER transition
        #     her_state, her_reward_py_float = self._create_her_transition(state, idx) # `idx` here is problematic for current _sample_her_goal
        #     if her_state is not None:
        #         # Convert Python float reward to a tensor of the buffer's dtype
        #         her_reward_tensor = torch.tensor(her_reward_py_float, dtype=self.dtype)
        #         return her_state, action, her_reward_tensor, done, next_state, extra_info

        # If HER is not applied (or disabled), return the original transition (reward_tensor is already a tensor)
        return state, action, reward_tensor, done, next_state, extra_info

    def _extract_goals(self, state):
        """Extract achieved and desired goals from flattened observation."""
        # Assuming observation structure: [obs(obs_dim), achieved_goal(goal_dim), desired_goal(goal_dim)]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy().astype(np.float32)  # Ensure float32
        elif isinstance(state, np.ndarray):
            state = state.astype(np.float32)  # Ensure float32

        total_expected = self.obs_dim + 2 * self.goal_dim
        if len(state) != total_expected:
            print(f"Warning: Expected obs length {total_expected}, got {len(state)}")
            # Fallback: assume equal split for goals
            goal_start = len(state) - 2 * self.goal_dim
            achieved_goal = state[goal_start : goal_start + self.goal_dim].astype(
                np.float32
            )
            desired_goal = state[goal_start + self.goal_dim :].astype(np.float32)
        else:
            achieved_goal = state[self.obs_dim : self.obs_dim + self.goal_dim].astype(
                np.float32
            )
            desired_goal = state[self.obs_dim + self.goal_dim :].astype(np.float32)

        return achieved_goal, desired_goal

    def _create_her_transition(self, original_state, idx):
        """Create HER transition by substituting goal."""
        if (
            len(self.current_episode) == 0
        ):  # This check is also tied to current_episode, problematic for general idx
            return None, None

        # Get new goal based on strategy
        # --- OPTION C NOTE for _sample_her_goal ---
        # If on-the-fly HER in __getitem__ were enabled and corrected, _sample_her_goal
        # would need to receive the *episode data corresponding to idx*, not rely on self.current_episode.
        new_goal = (
            self._sample_her_goal()
        )  # This call would need to pass episode context for idx
        if new_goal is None:
            return None, None

        # Create new state with substituted goal
        her_state = self._substitute_goal(original_state, new_goal)

        # Compute new reward
        achieved_goal, _ = self._extract_goals(original_state)
        her_reward = self.reward_fn(achieved_goal, new_goal)

        return (
            her_state,
            her_reward,
        )  # her_reward is already a Python float from _default_reward_fn

    def _sample_her_goal(self):
        """Sample a new goal based on HER strategy."""
        # --- OPTION C NOTE ---
        # For "future", "final", "episode" strategies to work correctly when called
        # from an on-the-fly __getitem__ relabeling for an arbitrary `idx`, this function
        # would need access to the specific episode from which `idx` was sampled.
        # Currently, it uses `self.current_episode` or `self.episodes` which might not
        # be the correct context for a transition `idx` from an older, completed episode.

        if self.her_strategy == "future" and len(self.current_episode) > 1:
            # Sample random future achieved goal from current episode
            future_transitions = self.current_episode[1:]  # Exclude first transition
            if future_transitions:
                transition = random.choice(future_transitions)
                return transition["achieved_goal"].astype(np.float32)

        elif self.her_strategy == "final" and len(self.current_episode) > 0:
            # Use final achieved goal from current episode
            return self.current_episode[-1]["achieved_goal"].astype(np.float32)

        elif self.her_strategy == "episode" and len(self.current_episode) > 0:
            # Sample any achieved goal from current episode
            transition = random.choice(self.current_episode)
            return transition["achieved_goal"].astype(np.float32)

        elif self.her_strategy == "random" and len(self.episodes) > 0:
            # Sample random achieved goal from any stored episode
            episode = random.choice(self.episodes)
            transition = random.choice(episode)
            return transition["achieved_goal"].astype(np.float32)

        return None

    def _substitute_goal(self, original_state, new_goal):
        """Substitute the desired goal in the observation."""
        if isinstance(original_state, torch.Tensor):
            state_np = original_state.cpu().numpy().astype(np.float32)
            is_tensor = True
        else:
            state_np = original_state.copy().astype(np.float32)
            is_tensor = False

        # Ensure new_goal is float32
        new_goal = np.asarray(new_goal, dtype=np.float32)

        # Replace desired_goal part with new_goal
        goal_start = self.obs_dim + self.goal_dim  # Start of desired_goal
        state_np[goal_start : goal_start + self.goal_dim] = new_goal

        if is_tensor:
            return torch.from_numpy(state_np).float()
        else:
            # Convert numpy array to tensor with correct dtype
            return torch.from_numpy(state_np).float()

    def _process_episode_her(self):
        """Process completed episode and add HER transitions."""
        if len(self.current_episode) < 2:
            return

        # Calculate the number of additional HER samples (k_to_add) based on her_ratio.
        # The relationship is her_ratio = k / (1+k), so k = her_ratio / (1-her_ratio).
        if self.her_ratio <= 0:
            k_to_add = 0
        elif (
            self.her_ratio >= 1.0
        ):  # Treat ratio >= 1.0 as wanting a high number of HER samples.
            # k_to_add = 19 aims for an effective HER ratio of 19/(1+19) = 95%.
            k_to_add = 19
        else:  # 0 < self.her_ratio < 1.0
            # Ensure denominator is not zero if her_ratio is pathologically close to 1.0
            denominator = 1.0 - self.her_ratio
            k_to_add = int(round(self.her_ratio / denominator))

        # Calculate the number of loops for adding HER transitions.
        # This is capped by the number of available future transitions possible
        num_loops_for_her = min(k_to_add, len(self.current_episode) - 1)

        # Add additional HER transitions for this episode
        for i, transition in enumerate(
            self.current_episode[:-1]
        ):  # Exclude last transition
            # Sample future goals and add HER transitions
            future_transitions = self.current_episode[i + 1 :]

            if not future_transitions:  # No future transitions to sample from
                continue

            # Add multiple HER transitions per step
            for _ in range(num_loops_for_her):  # Use the new calculated number of loops
                future_transition = random.choice(future_transitions)
                new_goal = future_transition["achieved_goal"].astype(np.float32)

                # Create HER state and reward
                her_state = self._substitute_goal(transition["state"], new_goal)
                her_reward = self.reward_fn(transition["achieved_goal"], new_goal)

                # Add HER transition to buffer - her_reward is already a Python float
                super().add(
                    her_state, transition["action"], her_reward, transition["done"]
                )

    def _default_reward_fn(self, achieved_goal, desired_goal):
        """Default reward function: 0 if close to goal, -1 otherwise."""
        # Ensure inputs are float32 numpy arrays
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)

        distance = np.linalg.norm(achieved_goal - desired_goal)
        threshold = 0.05  # 5cm threshold for success

        # Return Python float, not numpy scalar
        return float(0.0 if distance < threshold else -1.0)


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
        for idx, priority in zip(idcs, priorities, strict=False):
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
