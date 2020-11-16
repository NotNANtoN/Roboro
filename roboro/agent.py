import argparse

import torch

from roboro.policies import Q, QV
from roboro.networks import CNN


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
    def create_policy(obs_shape, act_shape, use_QV):
        if use_QV:
            return QV(obs_shape, act_shape)
        else:
            return Q(obs_shape, act_shape)

    def __init(self, observation_space, action_space,
               eps_start: float = 1.0,
               eps_end: float = 0.02,
               eps_last_frame: int = 150000,
               gamma: float = 0.99,
               qv: bool = False,
               ):
        obs_shape = self.get_net_obs_shape(observation_space)
        act_shape = self.get_net_act_shape(action_space)
        # Create feature extraction network
        self.obs_feature_net = CNN(obs_shape)
        obs_feature_shape = self.obs_feature_net.get_out_size()
        # Create policy networks:
        self.policy = self.create_policy(obs_feature_shape, act_shape, use_QV=qv)
        # Set hyperparams
        self.gamma = gamma

    def update_self(self, steps):
        """
        Update epsilon, target nets etc
        """
        pass

    def forward(self, obs):
        """Receives action preferences by policy and decides based off that"""
        features = self.obs_feature_net(obs)
        actions = self.policy(features)
        return actions

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        obs_feats = self.obs_feature_net(obs)
        next_obs_feats = self.obs_feature_net(next_obs)
        loss = self.policy.calc_loss(obs_feats, actions, rewards, done_flags, next_obs_feats, extra_info)
        return loss
