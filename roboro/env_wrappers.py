import itertools  # For ActionDiscretizerWrapper
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch

try:
    import aigar
except ModuleNotFoundError:
    print("Aigar envs could not be loaded")

# Register gymnasium-robotics environments
try:
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    print("Gymnasium-robotics environments registered successfully")
except ModuleNotFoundError:
    print("Gymnasium-robotics could not be loaded - Fetch environments unavailable")

from roboro.utils import apply_rec_to_dict, apply_to_state_list

atari_env_names = [
    "adventure",
    "airraid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank_heist",
    "battlezone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "choppercommand",
    "crazyclimber",
    "defender",
    "demonattack",
    "doubledunk",
    "elevatoraction",
    "enduro",
    "fishingderby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "icehockey",
    "jamesbond",
    "journey_escape",
    "kangaroo",
    "krull",
    "kungfumaster",
    "montezuma_revenge",
    "ms_pacman",
    "namethisgame",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "spaceinvaders",
    "stargunner",
    "tennis",
    "timepilot",
    "tutankham",
    "upndown",
    "venture",
    "videopinball",
    "wizardofwor",
    "yars_revenge",
    "zaxxon",
]


class ActionDiscretizerWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins_per_dim):
        super().__init__(env)
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "ActionDiscretizerWrapper only works with Box action spaces."

        self.num_bins_per_dim = num_bins_per_dim
        self.original_action_space = env.action_space
        self.action_dims = self.original_action_space.shape[0]

        if self.action_dims == 0:  # Scalar action space
            self.action_dims = 1
            self.low = np.array([self.original_action_space.low])
            self.high = np.array([self.original_action_space.high])
        else:
            self.low = self.original_action_space.low
            self.high = self.original_action_space.high

        self.num_discrete_actions = self.num_bins_per_dim**self.action_dims
        self.action_space = gym.spaces.Discrete(self.num_discrete_actions)

        # Create a map from discrete action index to continuous action values
        self.discrete_to_continuous_map = []

        # Generate all combinations of bin choices for each dimension
        # E.g., if action_dims=2, num_bins_per_dim=3
        # choices_per_dim will be [[0,1,2], [0,1,2]]
        choices_per_dim = [
            list(range(self.num_bins_per_dim)) for _ in range(self.action_dims)
        ]

        # itertools.product will give [(0,0), (0,1), (0,2), (1,0), ..., (2,2)]
        for choice_combination in itertools.product(*choices_per_dim):
            continuous_action = np.zeros(
                self.action_dims, dtype=self.original_action_space.dtype
            )
            for i in range(self.action_dims):
                dim_low = self.low[i]
                dim_high = self.high[i]
                if self.num_bins_per_dim == 1:
                    continuous_action[i] = (dim_low + dim_high) / 2.0
                else:
                    continuous_action[i] = dim_low + (
                        choice_combination[i] / (self.num_bins_per_dim - 1)
                    ) * (dim_high - dim_low)
            self.discrete_to_continuous_map.append(continuous_action)

    def action(self, discrete_action):
        if not (0 <= discrete_action < self.num_discrete_actions):
            raise ValueError(
                f"Discrete action {discrete_action} is out of bounds for {self.num_discrete_actions} actions."
            )
        continuous_action = self.discrete_to_continuous_map[discrete_action]
        # Ensure it's squeezed if the original action space was scalar but we made it 1D
        if self.original_action_space.shape == ():
            return continuous_action[0]
        return continuous_action


class VideoRecordingWrapper(gym.Wrapper):
    """Records videos of episodes in webm format and saves them to Weights & Biases."""

    def __init__(
        self,
        env,
        video_folder="videos",
        record_freq=0.05,
        video_length=30 * 10,
        fps=30,
        total_training_steps: int | None = None,
        warm_start_steps: int | None = None,
    ):
        """
        Args:
            env: The environment to wrap
            video_folder: Folder to store videos in
            record_freq: Frequency of recording (0.05 means record every 5% of total training steps)
            video_length: Maximum length of video in frames
            fps: Frames per second for the output video
            total_training_steps: Total number of training steps, used for percentage-based recording
            warm_start_steps: Number of initial warm-up steps to skip before starting to record
        """
        super().__init__(env)
        self.video_folder = video_folder
        self.record_freq = record_freq
        self.video_length = video_length
        self.fps = fps
        self.recording = False
        self.frames = []
        self.episode_reward = 0.0  # Track total reward for recorded episode
        self.total_steps = 0  # Track total steps taken
        self.last_recording_step = 0  # Track when we last recorded
        self.total_training_steps = total_training_steps
        self.warm_start_steps = warm_start_steps or 0  # Default to 0 if not provided
        self.num_milestones_passed = 0
        self.milestone_interval_steps = 0

        if (
            self.total_training_steps is not None
            and self.total_training_steps > 0
            and self.record_freq > 0
        ):
            # Adjust total_training_steps to account for warm-up phase
            effective_training_steps = self.total_training_steps - self.warm_start_steps
            self.milestone_interval_steps = int(
                self.record_freq * effective_training_steps
            )
            if self.milestone_interval_steps == 0 and effective_training_steps > 0:
                self.milestone_interval_steps = 1

        # Create video directory if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)

        # Import moviepy for video creation
        self.ImageSequenceClip = None
        self.image_sequence_clip_imported_successfully = False
        try:
            from moviepy import ImageSequenceClip

            self.ImageSequenceClip = ImageSequenceClip
            self.image_sequence_clip_imported_successfully = True
        except ImportError:
            print(
                "Warning: moviepy.ImageSequenceClip not found. Please install moviepy (e.g., 'pip install moviepy') for video recording."
            )

    def should_record_episode(self):
        """Determine if we should record this episode based on frequency."""
        # Never record during warm-up phase
        if self.total_steps < self.warm_start_steps:
            return False

        if self.total_training_steps is not None and self.milestone_interval_steps > 0:
            # Percentage-based recording using steps (after warm-up)
            if self.num_milestones_passed * self.milestone_interval_steps >= (
                self.total_training_steps - self.warm_start_steps
            ):
                return False
            current_milestone_target = (
                self.warm_start_steps
                + (self.num_milestones_passed + 1) * self.milestone_interval_steps
            )
            return self.total_steps >= current_milestone_target
        else:
            # Fallback to recording every N steps if total_training_steps not provided (still after warm-up)
            if self.record_freq <= 0:
                return False
            steps_since_last = self.total_steps - max(
                self.last_recording_step, self.warm_start_steps
            )
            steps_needed = int(1.0 / self.record_freq)
            return steps_since_last >= steps_needed

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1  # Increment step counter

        # Track episode reward if we're recording
        if self.recording:
            self.episode_reward += reward

        if self.recording and self.image_sequence_clip_imported_successfully:
            try:
                # Get RGB frame from environment
                frame = self.env.render()
                if frame is not None:
                    self.frames.append(frame)

                # Stop recording if episode ends or max length reached
                if terminated or truncated or len(self.frames) >= self.video_length:
                    self._save_video()
                    self.recording = False
                    self.frames = []  # Clear frames to free memory
            except Exception as e:
                print(f"Warning: Failed to record frame: {e}")
                self.recording = False
                self.frames = []

        # Check if we should start recording at the next step
        # This ensures we capture full episodes when we hit milestones
        if not self.recording and self.should_record_episode():
            self.recording = True
            self.episode_reward = 0.0  # Reset episode reward for new recording
            if (
                self.total_training_steps is not None
                and self.milestone_interval_steps > 0
            ):
                self.num_milestones_passed += 1
            else:
                self.last_recording_step = self.total_steps
            self.frames = []
            # Get initial frame for this step
            try:
                frame = self.env.render()
                if frame is not None:
                    self.frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to record initial frame: {e}")
                self.recording = False

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset episode reward when starting a new episode
        if self.recording:
            self.episode_reward = 0.0

        # If we're recording, get the initial frame
        if self.recording and self.image_sequence_clip_imported_successfully:
            try:
                frame = self.env.render()
                if frame is not None:
                    self.frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to record initial frame: {e}")
                self.recording = False

        return obs, info

    def _save_video(self):
        """Save the recorded frames as a webm video."""
        if not self.frames or not self.image_sequence_clip_imported_successfully:
            return

        try:
            # Convert frames to clip
            clip = self.ImageSequenceClip(self.frames, fps=self.fps)

            # Save as webm with quality settings for good compression
            video_path = os.path.join(
                self.video_folder, f"episode_{self.total_steps}.webm"
            )
            clip.write_videofile(
                video_path,
                codec="libvpx",
                audio=False,
                ffmpeg_params=[
                    "-crf",
                    "23",
                    "-b:v",
                    "0",
                ],  # Better quality/size balance
            )

            # Log to Weights & Biases
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "videos/episode_video": wandb.Video(
                                video_path, fps=self.fps, format="webm"
                            ),
                            "videos/episode_length": len(self.frames),
                            "videos/episode_step": self.total_steps,
                            "videos/episode_reward": self.episode_reward,
                        }
                    )
            except ImportError:
                print("Warning: wandb not found. Videos will only be saved locally.")
            except Exception as e:
                print(f"Warning: Failed to log video to wandb: {e}")

            # Close clip to free memory
            clip.close()

        except Exception as e:
            print(f"Warning: Failed to save video: {e}")

        # Clear frames to free memory regardless of success/failure
        self.frames = []


class TensorToNumpyActionWrapper(gym.ActionWrapper):
    """Converts torch tensor actions to numpy arrays or scalars as expected by the base environment."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        # If action is a torch tensor, convert it to numpy
        if hasattr(action, "detach"):  # Check if it's a tensor
            action = action.detach().cpu().numpy()

            # If it's a scalar action space, extract the scalar value
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = int(action.item()) if hasattr(action, "item") else int(action)
            elif (
                isinstance(self.action_space, gym.spaces.Box)
                and self.action_space.shape == ()
            ):
                action = (
                    float(action.item()) if hasattr(action, "item") else float(action)
                )

        return action


def create_env(
    env_name,
    frameskip,
    frame_stack,
    grayscale,
    sticky_action_prob,
    discretize_actions=False,
    num_bins_per_dim=5,
    CustomWrapper=None,  # noqa: N803
    render_mode: str | None = None,
    total_training_steps: int | None = None,
    warm_start_steps: int | None = None,
):
    # Init env:
    env = gym.make(
        env_name, render_mode="rgb_array"
    )  # Always use rgb_array for video recording

    # Handle dictionary observations from multi-goal environments (like FetchReach)
    if hasattr(env.observation_space, "spaces") or isinstance(
        env.observation_space, gym.spaces.Dict
    ):
        env = DictObsWrapper(env)

    # Apply Action Discretizer if configured and applicable
    if discretize_actions and isinstance(env.action_space, gym.spaces.Box):
        print(
            f"Discretizing continuous action space for {env_name} with {num_bins_per_dim} bins per dimension."
        )
        env = ActionDiscretizerWrapper(env, num_bins_per_dim)
        print(f"Number of discrete actions: {env.num_discrete_actions}")

    # Apply other Wrappers:
    if CustomWrapper is not None:
        env = CustomWrapper(env)
    if any(atari_name in env_name.lower() for atari_name in atari_env_names):
        env = AtariObsWrapper(env)
    if grayscale:
        if len(env.observation_space.shape) == 3:
            env = ToGrayScale(env)
        else:
            print(
                "Warning: Attempted to apply Grayscale wrapper to env without RGB space! Wrapper skipped."
            )
    env = ToTensor(env)
    if frameskip > 1:
        env = FrameSkip(env, skip=frameskip)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if sticky_action_prob > 0:
        env = StickyActions(env, sticky_action_prob)

    # Add tensor to numpy action conversion wrapper - this should be near the end
    # but before video recording to ensure actions are properly converted
    env = TensorToNumpyActionWrapper(env)

    # Add video recording wrapper
    env = VideoRecordingWrapper(
        env,
        total_training_steps=total_training_steps,
        warm_start_steps=warm_start_steps,
    )

    obs = env.reset()
    return env, obs


class ToGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dtype = None
        self.mean_dim = None

        self.setup(env.observation_space)
        if isinstance(env.observation_space, dict):
            new_space = apply_rec_to_dict(
                self.transform_obs_space, env.observation_space
            )
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
    permute dimensions of tensor to have channels first and convert to int8 (byte) data type.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 80, 80), dtype=env.observation_space.dtype
        )

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
        obs = None  # Initialize obs
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StickyActions(gym.Wrapper):
    """With a small probability, this wrapper applies the current action twice to the env."""

    def __init__(self, env, prob=0.25):
        super().__init__(env)
        self._prob = prob
        self._last_action = None

    def step(self, action):
        # repeat action with a small probability
        if self._last_action is not None and random.random() < self._prob:
            action = self._last_action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action  # Store the action that was actually taken
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)  # Pass kwargs for Gymnasium compatibility


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
            new_space = apply_rec_to_dict(
                self.transform_obs_space, env.observation_space
            )
            self.observation_space = dict(new_space)
        else:
            self.observation_space = self.transform_obs_space(self.observation_space)

    def transform_obs_space(self, obs_space):
        shp = list(obs_space.shape)
        shp[0] = shp[0] * self.k
        obs_space = gym.spaces.Box(low=0, high=1, shape=shp, dtype=obs_space.dtype)
        return obs_space

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)  # Gymnasium reset returns obs, info
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info  # Return obs, info

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
        # frames = self._frames
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


class DictObsWrapper(gym.ObservationWrapper):
    """Wrapper to handle dictionary observations from multi-goal environments.

    Flattens dictionary observations into a single vector by concatenating:
    - observation: robot state
    - achieved_goal: current end-effector position
    - desired_goal: target position
    """

    def __init__(self, env):
        super().__init__(env)

        # Get sample observation to determine structure
        sample_obs = env.observation_space.sample()
        if isinstance(sample_obs, dict):
            self.is_dict_obs = True
            self.dict_keys = list(sample_obs.keys())

            # Calculate total flattened size
            total_size = 0
            for key in self.dict_keys:
                total_size += np.prod(sample_obs[key].shape)

            # Create new flat Box observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
            )
            print(
                f"DictObsWrapper: Flattening dict obs {sample_obs.keys()} to shape {self.observation_space.shape}"
            )
        else:
            self.is_dict_obs = False
            print("DictObsWrapper: Non-dict observation, wrapper has no effect")

    def observation(self, obs):
        if not self.is_dict_obs:
            return obs

        # Flatten dictionary observation
        flat_obs = []
        for key in self.dict_keys:
            flat_obs.append(obs[key].flatten())

        return np.concatenate(flat_obs, dtype=np.float32)
