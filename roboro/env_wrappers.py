import random
from collections import deque
import itertools # For ActionDiscretizerWrapper

import gymnasium as gym
import torch
import numpy as np
try:
    import aigar
except ModuleNotFoundError:
    print("Aigar envs could not be loaded")

from roboro.utils import apply_rec_to_dict, apply_to_state_list


atari_env_names = ['adventure', 'airraid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
              'bank_heist', 'battlezone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
              'centipede', 'choppercommand', 'crazyclimber', 'defender', 'demonattack', 'doubledunk',
              'elevatoraction', 'enduro', 'fishingderby', 'freeway', 'frostbite', 'gopher', 'gravitar',
              'hero', 'icehockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kungfumaster',
              'montezuma_revenge', 'ms_pacman', 'namethisgame', 'phoenix', 'pitfall', 'pong', 'pooyan',
              'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
              'solaris', 'spaceinvaders', 'stargunner', 'tennis', 'timepilot', 'tutankham', 'upndown',
              'venture', 'videopinball', 'wizardofwor', 'yars_revenge', 'zaxxon']


class ActionDiscretizerWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins_per_dim):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "ActionDiscretizerWrapper only works with Box action spaces."
        
        self.num_bins_per_dim = num_bins_per_dim
        self.original_action_space = env.action_space
        self.action_dims = self.original_action_space.shape[0]

        if self.action_dims == 0: # Scalar action space
             self.action_dims = 1
             self.low = np.array([self.original_action_space.low])
             self.high = np.array([self.original_action_space.high])
        else:
            self.low = self.original_action_space.low
            self.high = self.original_action_space.high

        self.num_discrete_actions = self.num_bins_per_dim ** self.action_dims
        self.action_space = gym.spaces.Discrete(self.num_discrete_actions)

        # Create a map from discrete action index to continuous action values
        self.discrete_to_continuous_map = []
        
        # Generate all combinations of bin choices for each dimension
        # E.g., if action_dims=2, num_bins_per_dim=3
        # choices_per_dim will be [[0,1,2], [0,1,2]]
        choices_per_dim = [list(range(self.num_bins_per_dim)) for _ in range(self.action_dims)]
        
        # itertools.product will give [(0,0), (0,1), (0,2), (1,0), ..., (2,2)]
        for choice_combination in itertools.product(*choices_per_dim):
            continuous_action = np.zeros(self.action_dims, dtype=self.original_action_space.dtype)
            for i in range(self.action_dims):
                dim_low = self.low[i]
                dim_high = self.high[i]
                if self.num_bins_per_dim == 1:
                    continuous_action[i] = (dim_low + dim_high) / 2.0
                else:
                    continuous_action[i] = dim_low + (choice_combination[i] / (self.num_bins_per_dim - 1)) * (dim_high - dim_low)
            self.discrete_to_continuous_map.append(continuous_action)

    def action(self, discrete_action):
        if not (0 <= discrete_action < self.num_discrete_actions):
            raise ValueError(f"Discrete action {discrete_action} is out of bounds for {self.num_discrete_actions} actions.")
        continuous_action = self.discrete_to_continuous_map[discrete_action]
        # Ensure it's squeezed if the original action space was scalar but we made it 1D
        if self.original_action_space.shape == ():
            return continuous_action[0]
        return continuous_action


def create_env(env_name, frameskip, frame_stack, grayscale, sticky_action_prob, 
               discretize_actions=False, num_bins_per_dim=5, CustomWrapper=None):
    # Init env:
    env = gym.make(env_name)

    # Apply Action Discretizer if configured and applicable
    if discretize_actions and isinstance(env.action_space, gym.spaces.Box):
        print(f"Discretizing continuous action space for {env_name} with {num_bins_per_dim} bins per dimension.")
        env = ActionDiscretizerWrapper(env, num_bins_per_dim)

    # Apply other Wrappers:
    if CustomWrapper is not None:
        env = CustomWrapper(env)
    if any(atari_name in env_name.lower() for atari_name in atari_env_names):
        env = AtariObsWrapper(env)
    if grayscale:
        if len(env.observation_space.shape) == 3:
            env = ToGrayScale(env)
        else:
            print("Warning: Attempted to apply Grayscale wrapper to env without RGB space! Wrapper skipped.")
    env = ToTensor(env)
    if frameskip > 1:
        env = FrameSkip(env, skip=frameskip)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if sticky_action_prob > 0:
        env = StickyActions(env, sticky_action_prob)
    obs = env.reset()
    return env, obs


class ToGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dtype = None
        self.mean_dim = None

        self.setup(env.observation_space)
        if isinstance(env.observation_space, dict):
            new_space = apply_rec_to_dict(self.transform_obs_space, env.observation_space)
            self.observation_space = dict(new_space)
        else:
            self.observation_space = self.transform_obs_space(self.observation_space)

    def transform_obs_space(self, obs_space):
        shp = list(obs_space.shape)
        shp[self.mean_dim] = 1
        obs_space = gym.spaces.Box(shape=shp, dtype=obs_space.dtype, low=0, high=1)
        return obs_space

    def setup(self, obs_space):
        """Select dimension over which to take the mean"""
        obs_shape = obs_space.shape
        self.dtype = obs_space.dtype
        if obs_shape[0] == 3:
            self.mean_dim = 0
        elif obs_shape[-1] == 3:
            self.mean_dim = -1
        else:
            raise ValueError("Observation is not an RGB image!")

    def observation(self, obs):
        obs = obs.mean(axis=self.mean_dim)
        if self.dtype == np.uint8:
            obs = np.round(obs)
        obs = np.expand_dims(obs.astype(self.dtype), axis=0)
        return obs


class ToTensor(gym.ObservationWrapper):
    def observation(self, obs):
        return torch.from_numpy(np.ascontiguousarray(obs))


class AtariObsWrapper(gym.ObservationWrapper):
    """Cut out a 80x80 square, kill object flickering by taking the max of all pixels between two consecutive frames,
    permute dimensions of tensor to have channels first and convert to int8 (byte) data type."""
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 80, 80), dtype=env.observation_space.dtype)

    def observation(self, obs):
        # cut off score section
        obs = obs[35:195]
        # simple subsampling
        obs = obs[::2, ::2]
        # kill object flickering:
        if self.last_obs is not None:
            obs = np.max((self.last_obs, obs), axis=0)
        # put channels in first dim
        obs = np.moveaxis(obs, -1, 0)
        obs = obs.astype(np.uint8)
        return obs

    def reset(self):
        self.last_obs = None
        return super().reset()


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        assert skip > 0
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None # Initialize obs
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StickyActions(gym.Wrapper):
    """With a small probability, this wrapper applies the current action twice to the env.
    """
    def __init__(self, env, prob=0.25):
        super().__init__(env)
        self._prob = prob
        self._last_action = None

    def step(self, action):
        # repeat action with a small probability
        if self._last_action is not None and random.random() < self._prob:
            action = self._last_action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action # Store the action that was actually taken
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs) # Pass kwargs for Gymnasium compatibility


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

        if isinstance(env.observation_space, dict):
            new_space = apply_rec_to_dict(self.transform_obs_space, env.observation_space)
            self.observation_space = dict(new_space)
        else:
            self.observation_space = self.transform_obs_space(self.observation_space)

    def transform_obs_space(self, obs_space):
        shp = list(obs_space.shape)
        shp[0] = shp[0] * self.k
        obs_space = gym.spaces.Box(low=0, high=1, shape=shp, dtype=obs_space.dtype)
        return obs_space

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs) # Gymnasium reset returns obs, info
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info # Return obs, info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(tuple(self.frames))


class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to torch tensor before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.obs_is_dict = isinstance(self._frames[0], dict)

    def get_stacked_frames(self):
        import copy
        frames = copy.copy(self._frames)
        #frames = self._frames
        stacked = self._stack_frames(frames)
        return stacked

    def _stack_frames(self, frames):
        obs = apply_to_state_list(self._stack, frames)
        return obs

    @staticmethod
    def _stack(frames):
        return torch.cat(frames, dim=0)

    def to(self, *args, **kwargs):
        stacked = self.get_stacked_frames()
        if isinstance(args[0], torch.device) or isinstance(args[0], str):
            return stacked.to(*args, **kwargs)
        elif isinstance(args[0], torch.dtype):
            return stacked.to(dtype=args[0], **kwargs)
        return stacked.to(*args, **kwargs)

    def __array__(self, dtype=None):
        print("Access forbidden array")
        out = self.get_stacked_frames()
        if dtype is not None:
            out = out.type(dtype)
        return out

    def __len__(self):
        print("Access forbidden len")
        return len(self.get_stacked_frames())

    def __getitem__(self, i):
        print("Access forbidden getitem")
        return self._frames[i]

    def count(self):
        print("Access forbidden count")
        frames = self.get_stacked_frames()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        print("Access forbidden frame")
        return self.get_stacked_frames()[..., i]
