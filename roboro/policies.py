import torch

from roboro.networks import MLP


def polyak_update(net, target_net, factor):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(factor * target_param.data + param.data * (1.0 - factor))

def copy_weights(source_net, target_net):
        self.target_net.load_state_dict(self.source_net.state_dict())

def freeze_params(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class Q(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, gamma, dueling, noisy_layers, **net_kwargs):
        super().__init__()
        self.gamma = gamma
        self.q_net = MLP(obs_shape, act_shape, dueling=dueling, noisy=noisy_layers, **net_kwargs)
        self.q_net_target = MLP(obs_shape, act_shape, dueling=dueling, noisy=noisy_layers, **net_kwargs)
        freeze_params(self.q_net_target)

    def forward(self, obs):
        return self.q_net(obs)

    @torch.no_grad()
    def _calc_next_obs_q_vals(self, non_final_next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = torch.zeros(done_flags.shape, device=done_flags.device, dtype=non_final_next_obs.dtype)
        non_final_q_vals = net(non_final_next_obs)
        q_vals_next[~done_flags] = torch.max(non_final_q_vals, dim=1)[0]
        return q_vals_next

    def calc_target_val(self, rewards, done_flags, non_final_next_obs):
        q_vals_next = self._calc_next_obs_q_vals(non_final_next_obs, done_flags, self.q_net_target)
        assert q_vals_next.shape == rewards.shape
        targets = rewards + self.gamma * q_vals_next
        return targets

    def _get_q_preds(self, obs, actions):
        pred_q_vals = self.q_net(obs)
        # TODO: isn't here a better method than this stupid gather?
        chosen_action_q_vals = pred_q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        return chosen_action_q_vals

    def calc_loss(self, obs, actions, rewards, done_flags, non_final_next_obs, extra_info, targets=None):
        preds = self._get_q_preds(obs, actions)
        if targets is None:
            targets = self.calc_target_val(rewards, done_flags, non_final_next_obs)
        assert targets.shape == preds.shape
        loss = (targets - preds) ** 2
        return loss.mean()

    def update_target_nets_hard(self):
        copy_weights(self.q_net, self.q_net_target)

    def update_target_nets_soft(self, val):
        polyak_update(self.q_net, self.q_net_target, val)

class V(torch.nn.Module):
    """A state value network"""
    def __init__(self, obs_shape, act_shape, gamma, dueling, noisy_layers, **net_kwargs):
        super().__init__()
        self.gamma = gamma
        v_out_size = 1
        self.v_net = MLP(obs_shape, v_out_size, dueling=dueling, noisy=noisy_layers, **net_kwargs)
        self.v_net_target = MLP(obs_shape, v_out_size, dueling=dueling, noisy=noisy_layers, **net_kwargs)
        freeze_params(self.v_net_target)
        
    def forward(self, obs):
        return self.v_net(obs)

    @torch.no_grad()
    def _calc_next_obs_vals(self, non_final_next_obs, done_flags, net):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        v_vals_next = torch.zeros(done_flags.shape, device=done_flags.device, dtype=non_final_next_obs.dtype)
        v_vals_next[~done_flags] = net(non_final_next_obs).squeeze(1)
        return v_vals_next

    def calc_target_val(self, rewards, done_flags, next_obs):
        v_vals_next = self._calc_next_obs_vals(next_obs, done_flags, self.v_net_target)
        targets = rewards + self.gamma * v_vals_next
        return targets

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        preds = self.v_net(obs).squeeze()
        if targets is None:
            targets = self.calc_target_val(rewards, done_flags, next_obs)
        assert targets.shape == preds.shape, f'{preds.shape} - {targets.shape}'
        loss = (targets - preds) ** 2
        return loss.mean()

    def update_target_nets_hard(self):
        copy_weights(self.v_net, self.v_net_target)

    def update_target_nets_soft(self, val):
        polyak_update(self.v_net, self.v_net_target, val)


class QV(torch.nn.Module):
    """Train a state-action value network (Q) network and an additional state value network (V) and train the Q net using the V net"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q = Q(*args, **kwargs)
        self.v = V(*args, **kwargs)
        
    def forward(self, obs):
        return self.q(obs)
        
    def calc_loss(self, obs, actions, rewards, done_flags, non_final_next_obs, extra_info):
        v_target = self.v.calc_target_val(rewards, done_flags, non_final_next_obs)
        loss_args = (obs, actions, rewards, done_flags, non_final_next_obs, extra_info)
        v_loss = self.v.calc_loss(*loss_args, targets=v_target)
        q_loss = self.q.calc_loss(*loss_args, targets=v_target)
        loss = (v_loss + q_loss).mean()
        return loss
        
    def update_target_nets_hard(self):
        self.v.update_target_nets_hard()
        self.q.update_target_nets_hard()

    def update_target_nets_soft(self, val):
        self.v.update_target_nets_soft(val)
        self.q.update_target_nets_soft(val)


class Ensemble(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, PolicyClass):
        super().__init__(obs_shape)
        self.policies = [PolicyClass]

    def calc_loss(self, *args):
        losses = torch.stack([policy.calc_loss(*args) for policy in self.policies])
        loss = losses.mean()
        return loss
