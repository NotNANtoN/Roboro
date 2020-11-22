from collections import deque

import gym
import torch
import numpy as np

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


def create_env(env_name, frameskip, frame_stack, grayscale, CustomWrapper=None):
    # Init train_env:
    env = gym.make(env_name)

    # TODO: add Normalization wrapper in this function!
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
        env = FrameStack(env, frame_stack, stack_dim=0)
    obs = env.reset()
    return env, obs


class ToGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dtype = None
        self.mean_dim = None

    def setup(self, obs):
        assert obs.ndim == 4
        self.dtype = obs.dtype
        if obs.shape[1] == 3:
            self.mean_dim = 1
        elif obs.shape[-1] == 3:
            self.mean_dim = -1
        else:
            raise ValueError("Observation is not an RGB image!")

    def observation(self, obs):
        if self.mean_dim is None:
            self.setup(obs)
        obs = np.expand_dims(obs.mean(axis=self.mean_dim).astype(self.dtype), axis=0)
        return obs


class ToTensor(gym.ObservationWrapper):
    def observation(self, obs):
        # maybe np.ascontiguousarray() is needed before getting to torch
        return torch.from_numpy(obs).unsqueeze(0)


class AtariObsWrapper(gym.ObservationWrapper):
    """Cut out a 80x80 square, kill object flickering by taking the max of all pixels between two consecutive frames,
    convert to grayscale if needed, convert to torch tensor, permute dimensions of tensor to have channels first and
    convert to int8 (byte) data type."""
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 80, 80), dtype=env.observation_space.dtype)

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


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, stack_dim=0):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.stack_dim = stack_dim

        if isinstance(env.observation_space, dict):
            new_space = apply_rec_to_dict(self.transform_obs_space, env.observation_space)
            self.observation_space = dict(new_space)
        else:
            self.observation_space = self.transform_obs_space(self.observation_space)

        # The first dim of an incoming obs is the batch_size, don't stack it:
        self.stack_dim += 1

    def transform_obs_space(self, obs_space):
        shp = obs_space.shape
        stack_dim = self.stack_dim
        if stack_dim == -1:
            stack_dim = len(shp) - 1
        shp = [size * self.k if idx == stack_dim else size for idx, size in enumerate(shp)]
        obs_space = gym.spaces.Box(low=0, high=255, shape=shp, dtype=obs_space.dtype)
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
        return LazyFrames(tuple(self.frames), self.stack_dim)


class LazyFrames:
    def __init__(self, frames, stack_dim):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None
        self.obs_is_dict = isinstance(self._frames[0], dict)
        self.stack_dim = stack_dim

    def get_stacked_frames(self):
        return self.stack_frames(self._frames)

    def stack_frames(self, frames):
        obs = apply_to_state_list(self.stack, frames)
        return obs

    def stack(self, frames):
        return torch.cat(list(frames), dim=self.stack_dim)

    def make_state(self):
        return self.get_stacked_frames()

    def to(self, *args, **kwargs):
        return self.make_state().to(*args, **kwargs)

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
