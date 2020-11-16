import torch

from roboro.policies import Q, QV


class Agent(torch.nn.Module):
    """ Torch module that creates networks based off environment specifics, that returns actions based off states,
    and that calculates a loss based off an experience tuple
    """
    def __init(self, observation_space, action_space):
        obs_shape = self.get_net_obs_shape(observation_space)
        act_shape = self.get_net_act_shape(action_space)

        self.obs_feature_net = self.DQNCNN(obs_shape)
        obs_feature_shape = self.obs_feature_net.get_out_size()

        self.policy = self.create_policy(obs_feature_shape, act_shape, use_QV=False)

    @staticmethod
    def create_policy(self, obs_shape, act_shape, use_QV):
        if use_QV:
            return QV(obs_shape, act_shape)
        else:
            return Q(obs_shape, act_shape)

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
