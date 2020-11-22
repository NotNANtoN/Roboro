import torch

from roboro.networks import MLP


def polyak_update(net, target_net, factor):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(factor * target_param.data + param.data * (1.0 - factor))


class Q(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, gamma, **net_kwargs):
        super().__init__()
        self.gamma = gamma
        self.q_net = MLP(obs_shape, act_shape, **net_kwargs)
        self.q_net_target = MLP(obs_shape, act_shape, **net_kwargs)

    def forward(self, obs):
        q_vals = self.q_net(obs)
        action_preferences = torch.softmax(q_vals, dim=1)
        return action_preferences

    @torch.no_grad()
    def calc_next_obs_q_vals(self, non_final_next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = torch.zeros(done_flags.shape, device=done_flags.device)
        non_final_q_vals = net(non_final_next_obs)
        q_vals_next[~done_flags] = torch.max(non_final_q_vals, dim=1)[0]
        return q_vals_next

    def calc_q_target_val(self, rewards, done_flags, non_final_next_obs):
        q_vals_next = self.calc_next_obs_q_vals(non_final_next_obs, done_flags, self.q_net_target)
        assert q_vals_next.shape == rewards.shape
        targets = rewards + self.gamma * q_vals_next
        return targets

    def get_q_preds(self, obs, actions):
        pred_q_vals = self.q_net(obs)
        # TODO: isn't here a better method than this stupid gather?
        chosen_action_q_vals = pred_q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        return chosen_action_q_vals

    def mse_loss(self, obs, actions, rewards, done_flags, non_final_next_obs):
        preds = self.get_q_preds(obs, actions)
        target_q_vals = self.calc_q_target_val(rewards, done_flags, non_final_next_obs)
        assert target_q_vals.shape == preds.shape
        loss = (target_q_vals - preds) ** 2
        loss = loss.mean()
        return loss

    def calc_loss(self, obs, actions, rewards, done_flags, non_final_next_obs, extra_info):
        loss = self.mse_loss(obs, actions, rewards, done_flags, non_final_next_obs)
        return loss

    def update_target_nets_hard(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def update_target_nets_soft(self, val):
        polyak_update(self.q_net, self.q_net_target, val)


class QV(Q):
    """Train an additional state value network and train the q_net using it"""
    def __init__(self, obs_shape, act_shape, gamma, **net_kwargs):
        super().__init__(obs_shape, act_shape, gamma)
        v_out_size = 1
        self.v_net = MLP(obs_shape, v_out_size, **net_kwargs)
        self.v_net_target = MLP(obs_shape, v_out_size, **net_kwargs)

    @torch.no_grad()
    def calc_next_obs_v_vals(self, non_final_next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        v_vals_next = torch.zeros(done_flags.shape, device=done_flags.device)
        v_vals_next[~done_flags] = net(non_final_next_obs).squeeze(1)
        return v_vals_next

    def calc_v_target_val(self, rewards, done_flags, next_obs):
        v_vals_next = self.calc_next_obs_v_vals(next_obs, done_flags, self.v_net_target)
        targets = rewards + self.gamma * v_vals_next
        return targets

    def mse_loss(self, obs, actions, rewards, done_flags, next_obs):
        # V loss
        preds_v = self.v_net(obs)
        targets = self.calc_v_target_val(rewards, done_flags, next_obs)
        v_loss = (targets - preds_v) ** 2
        # Q loss
        preds_q = self.get_q_preds(obs, actions)
        # summed loss
        q_loss = (targets - preds_q) ** 2
        loss = (q_loss + v_loss).mean()
        return loss

    def update_target_nets_hard(self):
        super().update_target_nets_hard()
        self.v_net_target.load_state_dict(self.v_net.state_dict())

    def update_target_nets_soft(self, val):
        super().update_target_nets_soft(val)
        polyak_update(self.v_net, self.v_net_target, val)


class Ensemble(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, PolicyClass):
        super().__init__(obs_shape)
        self.policies = [PolicyClass]

    def calc_loss(self, *args):
        losses = torch.stack([policy.calc_loss(*args) for policy in self.policies])
        loss = losses.mean()
        return loss
