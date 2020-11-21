import argparse

import torch
import gym

from roboro.policies import Q, QV
from roboro.networks import CNN, MLP


def get_act_len(act_space):
    assert isinstance(act_space, gym.spaces.Discrete)
    return act_space.n


class Agent(torch.nn.Module):
    """ Torch module that creates networks based off environment specifics, that returns actions based off states,
    and that calculates a loss based off an experience tuple
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser

    @staticmethod
    def create_policy(obs_shape, act_shape, gamma, use_QV, **net_kwargs):
        if use_QV:
            return QV(obs_shape, act_shape, gamma, **net_kwargs)
        else:
            return Q(obs_shape, act_shape, gamma, **net_kwargs)

    def __init__(self, obs_sample, action_space,
                 eps_start: float = 0.1,
                 eps_end: float = 0.02,
                 eps_last_frame: int = 150000,
                 gamma: float = 0.99,
                 qv: bool = False,
                 layer_width: int = 256,
                 ):
        super().__init__()
        # Set hyperparams
        self.epsilon = eps_start
        # Get in-out shapes:
        obs_shape = obs_sample.shape  #self.get_net_obs_shape(obs_sample)
        act_shape = get_act_len(action_space)
        print("Obs shape: ", obs_shape, " Act shape: ", act_shape)
        # Create feature extraction network
        self.obs_feature_net = CNN(obs_shape) if len(obs_shape) == 3 else MLP(obs_shape[0], layer_width)
        obs_feature_shape = self.obs_feature_net.get_out_size()
        # Create policy networks:
        self.policy = self.create_policy(obs_feature_shape, act_shape, gamma, use_QV=qv, width=layer_width)

    def update_self(self, steps):
        """
        Update epsilon, target nets etc
        """
        pass

    def forward(self, obs):
        """Receives action preferences by policy and decides based off that"""
        features = self.obs_feature_net(obs)
        q_vals = self.policy(features)
        actions = torch.argmax(q_vals, dim=1)
        return actions

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        assert done_flags.dtype == torch.bool
        obs_feats = self.obs_feature_net(obs)
        next_obs_feats = self.obs_feature_net(next_obs[~done_flags])
        loss = self.policy.calc_loss(obs_feats, actions, rewards, done_flags, next_obs_feats, extra_info)
        # TODO: incorporate PER weigh tupdate in extra_info
        return loss, extra_info
