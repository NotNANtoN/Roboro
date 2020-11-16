import torch

from roboro.networks import MLP, CNN

class Q(torch.nn.Module):
    def __init__(self, obs_shape, act_shape):
        super().__init__()
        self.q_net = MLP(obs_shape, act_shape)
        self.q_net_target = MLP(obs_shape, act_shape)

    def forward(self, obs):
        q_vals = self.calc_vals(obs, self.q_net)
        action_preferences = torch.softmax(q_vals, dim=1)
        return action_preferences

    def calc_vals(self, obs, net):
        obs_features = self.obs_feature_net(obs)
        vals = net(obs_features)
        return vals

    @torch.no_grad()
    def calc_next_obs_q_vals(self, next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = torch.zeros(done_flags, device=done_flags.device)
        non_final_idcs = torch.where(done_flags)
        non_final_next_obs = next_obs[non_final_idcs]
        non_final_q_vals = self.calc_vals(non_final_next_obs, net)
        q_vals_next[non_final_idcs] = torch.max(non_final_q_vals, dim=1)[0]
        return q_vals_next

    def calc_q_target_val(self, rewards, done_flags, next_obs):
        q_vals_next = self.calc_next_obs_q_vals(next_obs, done_flags, self.q_net_target)
        target_q_vals = rewards + self.gamma * q_vals_next
        return target_q_vals

    def mse_loss_q(self, obs, actions, rewards, done_flags, next_obs):
        pred_q_vals = self.calc_vals(obs, self.q_net).gather(actions, dim=1)
        target_q_vals = self.calc_q_target(rewards, done_flags, next_obs)
        loss = (target_q_vals - pred_q_vals) ** 2
        return loss

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        loss = self.mse_loss_q(self, obs, actions, rewards, done_flags, next_obs)
        return loss


class QV(Q):
    """Train an additional state value network and train the q_net using it"""
    def __init__(self, obs_shape, act_shape):
        super().__init__(obs_shape, act_shape)
        v_out_shape = [1]
        self.v_net = MLP(obs_shape, v_out_shape)
        self.v_net_target = MLP(obs_shape, v_out_shape)

    @torch.no_grad()
    def calc_next_obs_v_vals(self, next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        v_vals_next = torch.zeros(done_flags, device=done_flags.device)
        non_final_idcs = torch.where(done_flags)
        non_final_next_obs = next_obs[non_final_idcs]
        v_vals_next[non_final_idcs] = self.calc_vals(non_final_next_obs, net)
        return v_vals_next

    def calc_v_target_val(self, rewards, done_flags, next_obs):
        v_vals_next = self.calc_next_obs_v_vals(next_obs, done_flags, self.v_net_target)
        target_v_vals = rewards + self.gamma * v_vals_next
        return target_v_vals

    def mse_loss_v(self, obs, rewards, done_flags, next_obs):
        pred_v_vals = self.calc_vals(obs, self.v_net)
        target_v_vals = self.calc_v_target_val(rewards, done_flags, next_obs)
        loss = (target_v_vals - pred_v_vals) ** 2
        return loss

    def calc_q_target(self, rewards, done_flags, next_obs):
        """Overwrite this method to force Q-net to take V-net targets"""
        return self.calc_v_target_val(rewards, done_flags, next_obs)

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_loss = self.mse_loss_q(obs, actions, rewards, done_flags, next_obs)
        v_loss = self.mse_loss_v(obs, rewards, done_flags, next_obs)
        return q_loss + v_loss


class Ensemble(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, PolicyClass):
        super().__init__(obs_shape)
        self.policies = [PolicyClass]

    def calc_loss(self, *args):
        losses = torch.stack([policy.calc_loss(*args) for policy in self.policies])
        loss = losses.mean()
        return loss
