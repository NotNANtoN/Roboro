import random

import numpy as np
import torch
import gym
from omegaconf import DictConfig

from roboro.policies import create_policy
from roboro.networks import CNN, MLP
from roboro.utils import Standardizer
from roboro.world_model import MuZero
from roboro.mcts import MCTS


def get_act_len(act_space):
    assert isinstance(act_space, gym.spaces.Discrete)
    return act_space.n


class Agent(torch.nn.Module):
    """ Torch module that creates networks based off environment specifics, that returns actions based off states,
    and that calculates a loss based off an experience tuple
    """
    def __init__(self, obs_space, action_space,
                 double_q: bool = False,
                 qv: bool = False,
                 qvmax: bool = False,
                 iqn: bool = False,
                 soft_q: bool = False,
                 munch_q: bool = False,
                 int_ens: bool = False,
                 rem: bool = False,

                 eps_start: float = 0.1,
                 target_net_hard_steps: int = 1000,
                 target_net_polyak_val: float = 0.99,
                 target_net_use_polyak: bool = True,
                 warm_start_steps: int = 1000,
                 feat_layer_width: int = 256,
                 policy: DictConfig = None,
                 ):
        super().__init__()
        # Set hyperparams
        self.epsilon = eps_start
        self.stored_epsilon = self.epsilon
        self.target_net_hard_steps = target_net_hard_steps
        self.target_net_polyak_val = target_net_polyak_val
        self.target_net_use_polyak = target_net_use_polyak
        # Get in-out shapes:
        obs_sample = obs_space.sample()
        obs_shape = obs_sample.shape
        self.obs_shape = obs_shape
        self.act_shape = get_act_len(action_space)
        print("Obs space: ", obs_space)
        print("Obs shape: ", obs_shape, " Act shape: ", self.act_shape)
        # Create feature extraction network
        self.normalizer = Standardizer(warm_start_steps)
        self.obs_feature_net = CNN(obs_shape, feat_layer_width) if len(obs_shape) == 3\
            else MLP(obs_shape[0], feat_layer_width)
        obs_feature_shape = self.obs_feature_net.get_out_size()
        # Create policy networks:
        self.policy = create_policy(obs_feature_shape, self.act_shape, double_q=double_q, rem=rem, int_ens=int_ens,
                                         use_qv=qv, use_qvmax=qvmax, iqn=iqn, use_soft_q=soft_q, use_munch_q=munch_q,
                                         policy_kwargs=policy, )
        print("Policy: ", self.policy)

    def update_self(self, steps):
        """
        Update epsilon, target nets etc
        """
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
        if random.random() < self.epsilon:
            q_vals = torch.rand(len(obs), self.act_shape)
        else:
            features = self.extract_features(obs)
            q_vals = self.policy(features)
        actions = torch.argmax(q_vals, dim=1)
        return actions

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        assert done_flags.dtype == torch.bool
        obs_feats = self.extract_features(obs)
        next_obs_feats = self.extract_features(next_obs)
        loss = self.policy.calc_loss(obs_feats, actions, rewards, done_flags, next_obs_feats, extra_info)
        # TODO: incorporate PER weight update in extra_info
        return loss, extra_info

    def eval(self):
        self.stored_epsilon = self.epsilon
        self.epsilon = 0

    def train(self, mode: bool = True):
        self.epsilon = self.stored_epsilon if self.stored_epsilon is not None else self.epsilon

class MuAgent(Agent):
    def __init__(self, obs_space, action_space,
                 double_q: bool = False,
                 qv: bool = False,
                 qvmax: bool = False,
                 iqn: bool = False,
                 soft_q: bool = False,
                 munch_q: bool = False,
                 int_ens: bool = False,
                 rem: bool = False,

                 eps_start: float = 0.1,
                 target_net_hard_steps: int = 1000,
                 target_net_polyak_val: float = 0.99,
                 target_net_use_polyak: bool = True,
                 warm_start_steps: int = 1000,
                 feat_layer_width: int = 256,
                 policy: DictConfig = None,

                 n_simulations: int = 50,
                 model_args: DictConfig = None,
                 ):
        Agent.__init__(self, obs_space, action_space, double_q, qv, qvmax,
                 iqn, soft_q, munch_q, int_ens, rem, eps_start,
                 target_net_hard_steps, target_net_polyak_val, target_net_use_polyak, warm_start_steps,
                 feat_layer_width, policy)

        self.world_model = MuZero(self.obs_shape, self.act_shape, self.normalizer, model_args)
        self.mcts = MCTS(model_args.mcts, action_space)

    @staticmethod
    def select_action(node, temperature):
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return torch.tensor(action)

    def forward(self, obs):
        self.normalizer.observe(obs)  # observe obs and update mean and std
        if random.random() < self.epsilon:
            self.q_vals = torch.rand(len(obs), self.act_shape)
            actions = torch.argmax(self.q_vals, dim=1)
        else:
            root, extra_info = self.mcts.run(self.world_model, obs)
            actions = self.select_action(root, 0.25)

            features = self.extract_features(obs)
            self.q_vals = self.policy(features)

        actions = torch.argmax(self.q_vals, dim=1)
        return actions
