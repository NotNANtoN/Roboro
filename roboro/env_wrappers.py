import random
from collections import deque

import gym
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


def create_env(env_name, frameskip, frame_stack, grayscale, sticky_action_prob, CustomWrapper=None):
    # Init env:
    env = gym.make(env_name)

    # Apply Wrappers:
    if CustomWrapper is not None:
        env = CustomWrapper(env)
    if any(atari_name in env_name.lower() for atari_name in atari_env_names):
        env = AtariObsWrapper(env)
    if grayscale:
        env = ToGrayScale(env)
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
        assert len(obs_shape) == 3
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
        obs = obs[35:195]
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
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class StickyActions(gym.Wrapper):
    """With a small probability, this wrapper applies the current action twice to the env.
    """
    def __init__(self, env, prob=0.25):
        super().__init__(env)
        self._prob = prob

    def step(self, action):
        obs, total_reward, done, info = self.env.step(action)
        # repeat action with a small probability
        if not done and random.random() < self._prob:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return obs, total_reward, done, info


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

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

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
        return self.get_stacked_frames().to(*args, **kwargs)

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
