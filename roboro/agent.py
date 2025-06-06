import random

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from roboro.networks import CNN, MLP
from roboro.policies import create_policy
from roboro.utils import Standardizer, map_bin_indices_to_continuous_tensor


def get_act_len(act_space):
    assert isinstance(act_space, gym.spaces.Discrete)
    return act_space.n


class Agent(torch.nn.Module):
    """Torch module that creates networks based off environment specifics, that returns actions based off states,
    and that calculates a loss based off an experience tuple
    """

    def __init__(
        self,
        obs_space,
        action_space,
        double_q: bool = False,
        qv: bool = False,
        qvmax: bool = False,
        iqn: bool = False,
        soft_q: bool = False,
        munch_q: bool = False,
        int_ens: bool = False,
        rem: bool = False,
        clip_rewards: bool = False,
        eps_start: float = 1.0,  # Updated default
        eps_end: float = 0.05,
        eps_decay_steps: int = 10000,
        target_net_hard_steps: int = 1000,
        target_net_polyak_val: float = 0.99,
        target_net_use_polyak: bool = True,
        warm_start_steps: int = 1000,
        feat_layer_width: int = 256,
        policy: DictConfig | None = None,
        # New parameters for independent dimension binning
        discretization_method: str = "joint",  # "joint" or "independent_dims"
        num_bins_per_dim: int = 5,  # Used if discretization_method is 'independent_dims'
    ):
        super().__init__()
        # Set hyperparams
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = self.eps_start  # Initialize epsilon
        self.stored_epsilon = self.epsilon  # For eval mode

        self.target_net_hard_steps = target_net_hard_steps
        self.target_net_polyak_val = target_net_polyak_val
        self.target_net_use_polyak = target_net_use_polyak
        self.clip_rewards = clip_rewards

        self.discretization_method = discretization_method
        self.num_bins_per_dim = num_bins_per_dim

        # Get in-out shapes:
        obs_sample = obs_space.sample()
        obs_shape = obs_sample.shape

        policy_creation_kwargs = dict(policy) if policy is not None else {}
        net_config = policy_creation_kwargs.get("net", {})
        activation_fn = net_config.get("activation_fn", "relu")
        use_layer_norm = net_config.get("use_layer_norm", False)

        if self.discretization_method == "independent_dims":
            assert isinstance(
                action_space, gym.spaces.Box
            ), "'independent_dims' discretization only works with Box action spaces."
            self.action_dims = action_space.shape[0]
            self.policy_act_shape = self.action_dims * self.num_bins_per_dim
            # Store action space bounds for mapping
            # These will be moved to the correct device in train_setup or similar
            self.register_buffer(
                "action_low", torch.from_numpy(action_space.low.astype(np.float32))
            )
            self.register_buffer(
                "action_high", torch.from_numpy(action_space.high.astype(np.float32))
            )
            policy_creation_kwargs["action_dims"] = self.action_dims
            policy_creation_kwargs["num_bins_per_dim"] = self.num_bins_per_dim
            policy_creation_kwargs["discretization_method"] = "independent_dims"

        elif self.discretization_method == "joint":
            assert isinstance(
                action_space, gym.spaces.Discrete
            ), "'joint' discretization expects a Discrete action space (env should be wrapped)."
            self.policy_act_shape = get_act_len(action_space)
            self.action_dims = None  # Not directly used by policy in this mode
            policy_creation_kwargs["discretization_method"] = "joint"
        else:
            raise ValueError(
                f"Unknown discretization_method: {self.discretization_method}"
            )

        print("Obs space: ", obs_space)
        print(
            f"Policy act shape: {self.policy_act_shape}, Method: {self.discretization_method}"
        )
        if self.action_dims:
            print(
                f"Action dims: {self.action_dims}, Bins per dim: {self.num_bins_per_dim}"
            )

        # Create feature extraction network
        self.normalizer = Standardizer(warm_start_steps)
        self.obs_feature_net = (
            CNN(obs_shape, feat_layer_width, activation_fn=activation_fn)
            if len(obs_shape) == 3
            else MLP(
                obs_shape[0],
                feat_layer_width,
                activation_fn=activation_fn,
                use_layer_norm=use_layer_norm,
            )
        )
        obs_feature_shape = self.obs_feature_net.get_out_size()
        # Create policy networks:
        self.policy = create_policy(
            obs_feature_shape,
            self.policy_act_shape,  # Use the calculated policy_act_shape
            double_q=double_q,
            rem=rem,
            int_ens=int_ens,
            use_qv=qv,
            use_qvmax=qvmax,
            iqn=iqn,
            use_soft_q=soft_q,
            use_munch_q=munch_q,
            policy_kwargs=policy_creation_kwargs,  # Pass updated kwargs
        )
        print("Policy: ", self.policy)

    def update_self(self, steps):
        """
        Update epsilon, target nets etc
        """
        # Epsilon decay (Linear)
        if self.eps_decay_steps > 0:
            fraction = min(1.0, float(steps) / self.eps_decay_steps)
            self.epsilon = self.eps_start + fraction * (self.eps_end - self.eps_start)
            self.epsilon = max(
                self.eps_end, self.epsilon
            )  # Ensure it doesn't go below eps_end

        if self.target_net_use_polyak:
            self.policy.update_target_nets_soft(self.target_net_polyak_val)
        elif steps % self.target_net_hard_steps == 0:
            self.policy.update_target_nets_hard()

    def extract_features(self, obs):
        obs = self.normalizer.norm(obs)
        features = self.obs_feature_net(obs)
        return features

    def forward(self, obs):
        """Receives action preferences by policy and decides based off that"""
        self.normalizer.observe(obs)  # observe obs and update mean and std
        features = self.extract_features(obs)
        q_vals_or_actions = self.policy(
            features
        )  # For joint, these are q_vals, for independent, these are q_vals_flat

        if self.discretization_method == "independent_dims":
            # q_vals_or_actions are flat Q-values for independent dims: (batch, action_dims * num_bins)
            q_values_structured = q_vals_or_actions.view(
                -1, self.action_dims, self.num_bins_per_dim
            )
            if random.random() < self.epsilon:
                # Explore: choose random bin for each dimension
                actions = torch.randint(
                    0,
                    self.num_bins_per_dim,
                    (obs.shape[0], self.action_dims),
                    device=obs.device,
                    dtype=torch.long,
                )
            else:
                # Exploit: choose best bin for each dimension
                actions = torch.argmax(
                    q_values_structured, dim=2
                )  # (batch, action_dims)
        elif self.discretization_method == "joint":
            # q_vals_or_actions are Q-values for joint discrete actions: (batch, num_discrete_actions)
            if random.random() < self.epsilon:
                # For joint, actions are single integers
                actions = torch.randint(
                    0, self.policy_act_shape, (obs.shape[0],), device=obs.device
                )
            else:
                actions = torch.argmax(q_vals_or_actions, dim=1)
        else:
            raise ValueError(
                f"Unknown discretization_method: {self.discretization_method}"
            )

        return actions  # Return chosen bin indices for independent_dims, or single action index for joint

    def _map_chosen_bin_indices_to_continuous_actions(
        self, chosen_bin_indices_batch: torch.Tensor
    ) -> torch.Tensor:
        """Maps a batch of chosen bin indices to continuous actions."""
        if self.discretization_method != "independent_dims":
            raise ValueError(
                "This method is only for 'independent_dims' discretization."
            )
        return map_bin_indices_to_continuous_tensor(
            chosen_bin_indices_batch,
            self.action_low,
            self.action_high,
            self.num_bins_per_dim,
        )

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        assert done_flags.dtype == torch.bool
        if self.clip_rewards:
            rewards = torch.sign(rewards)
        obs_feats = self.extract_features(obs)
        next_obs_feats = self.extract_features(next_obs)
        loss, tde = self.policy.calc_loss(
            obs_feats, actions, rewards, done_flags, next_obs_feats, extra_info
        )
        extra_info["tde"] = tde
        return loss, extra_info

    def eval(self):
        self.stored_epsilon = self.epsilon  # Store current epsilon before setting to 0
        self.epsilon = 0
        # Ensure action_low and action_high are on the same device as the model if used in eval
        if self.discretization_method == "independent_dims":
            # This assumes the model is already on its target device.
            # Buffers should ideally be moved with model.to(device)
            # For safety, ensure they are on a consistent device if used here, though
            # eval() itself doesn't directly use them, but good practice.
            # self.action_low = self.action_low.to(self.device) # self.device is not directly available
            # self.action_high = self.action_high.to(self.device)
            pass
        super().eval()

    def train(self, mode: bool = True):
        super().train(mode)  # It's good practice to call super().train() first
        if mode:
            self.epsilon = (
                self.stored_epsilon
                if self.stored_epsilon is not None
                else self.eps_start
            )
            # Ensure action_low and action_high are on the same device as the model
            if self.discretization_method == "independent_dims":
                # self.action_low = self.action_low.to(self.device)
                # self.action_high = self.action_high.to(self.device)
                pass  # Handled by PyTorch Lightning's automatic device placement for buffers
        else:  # If switching to eval mode (mode is False)
            self.stored_epsilon = (
                self.epsilon
            )  # Store current epsilon before setting to 0 for eval
            self.epsilon = 0
