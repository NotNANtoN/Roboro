import torch
from omegaconf import DictConfig

from roboro.networks import MLP, IQNNet
from roboro.utils import polyak_update, copy_weights, freeze_params, calculate_huber_loss


class Policy(torch.nn.Module):
    def __init__(self, gamma=None):
        """ Policy superclass that deals with basic and repetitiv tasks such as updating the target networks or
         calculating the gamma values.

         Subclasses need to define self.nets and self.target_nets such that the target nets are updated properly.
         """
        super().__init__()
        self.gamma = gamma
        self.nets = []
        self.target_nets = []

    def __str__(self):
        return f'Policy_{self.gamma}'

    def _calc_gammas(self, done_flags, extra_info):
        """Apply the discount factor. If a done flag is set we discount by 0."""
        gammas = (~done_flags).float().squeeze()
        if "n_step" in extra_info:
            gammas *= self.gamma ** extra_info["n_step"]
        else:
            gammas *= self.gamma
        return gammas

    def update_target_nets_hard(self):
        for net, target_net in zip(self.nets, self.target_nets):
            copy_weights(net, target_net)

    def update_target_nets_soft(self, val):
        for net, target_net in zip(self.nets, self.target_nets):
            polyak_update(net, target_net, val)

    def forward(self, obs):
        raise NotImplementedError

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        raise NotImplementedError


class MultiNetPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policies = []

    def update_target_nets_hard(self):
        for pol in self.policies:
            pol.update_target_nets_hard()

    def update_target_nets_soft(self, val):
        for pol in self.policies:
            pol.update_target_nets_soft(val)


class Q(Policy):
    def __init__(self, obs_size, act_size, net: DictConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_args = (obs_size, act_size)
        self.net_kwargs = net
        self.q_net = self.create_net()
        self.q_net_target = self.create_net(target=True)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

    def create_net(self, target=False):
        net = MLP(*self.net_args, **self.net_kwargs)
        if target:
            freeze_params(net)
        return net

    def __str__(self):
        return f'Q <{super().__str__()}>'

    def forward(self, obs):
        return self.q_net(obs)

    @torch.no_grad()
    def forward_target(self, obs):
        return self.q_net_target(obs)

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        preds = self._get_q_preds(obs, actions)
        if targets is None:
            targets = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        assert targets.shape == preds.shape, f"{targets.shape}, {preds.shape}"
        loss = (targets - preds) ** 2
        return loss.mean()

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_vals_next = self.next_obs_val(next_obs)
        assert q_vals_next.shape == rewards.shape
        gammas = self._calc_gammas(done_flags, extra_info)
        targets = rewards + gammas * q_vals_next
        return targets

    def obs_val(self, obs, net=None):
        if net is None:
            net = self.q_net
        return net(obs)

    def _get_q_preds(self, obs, actions):
        """Get Q-value predictions for current obs based on actions"""
        pred_q_vals = self.obs_val(obs)
        # TODO: isn't there a better method than this stupid gather?
        chosen_action_q_vals = pred_q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        return chosen_action_q_vals

    def next_obs_val(self, *args):
        """Calculate the value of the next obs via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        # Next state action selection
        max_idcs, q_vals_next = self.next_obs_act_select(*args)
        # Next state action evaluation
        q_vals_next = self.next_obs_act_eval(max_idcs, *args, q_vals_next_eval=q_vals_next)
        return q_vals_next

    def next_obs_act_select(self, next_obs, use_target_net=True):
        # Next state action selection
        q_vals_next = self.q_pred_next_state(next_obs, use_target_net=use_target_net)
        max_idcs = torch.argmax(q_vals_next, dim=1, keepdim=True)
        return max_idcs, q_vals_next

    def next_obs_act_eval(self, max_idcs, next_obs, q_vals_next_eval=None, use_target_net=True):
        if q_vals_next_eval is None:
            q_vals_next_eval = self.q_pred_next_state(next_obs, use_target_net=use_target_net)
        q_vals_next = q_vals_next_eval.gather(dim=1, index=max_idcs).squeeze()
        return q_vals_next

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, use_target_net=True, net=None):
        """ Specified in extra method to be potentially overridden by subclasses"""
        if net is None:
            if use_target_net:
                net = self.q_net_target
            else:
                net = self.q_net
        return net(next_obs)


class IQNQ(Policy):
    """
    IQN Policy that uses the IQN Layer and calculates a loss. Adapted from https://github.com/BY571/IQN-and-Extensions
    """
    def __init__(self, obs_size, act_size, *args, num_tau=16, num_policy_samples=32, net: DictConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_size = act_size
        self.huber_thresh = 1.0

        # Create IQN-Network
        self.net_args = (obs_size, act_size, num_tau, num_policy_samples)
        self.net_kwargs = net
        self.q_net = self.create_net()
        self.q_net_target = self.create_net(target=True)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

    def __str__(self):
        return f'IQN <{super().__str__()}>'

    def forward(self, obs):
        return self.q_net(obs)

    @torch.no_grad()
    def forward_target(self, obs):
        return self.q_net_target(obs)

    def create_net(self, target=False):
        net = IQNNet(*self.net_args, **self.net_kwargs)
        if target:
            freeze_params(net)
        return net

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        # Reshape:
        batch_size = obs.shape[0]
        rewards = rewards.unsqueeze(-1)
        actions = actions.unsqueeze(-1)
        done_flags = done_flags.unsqueeze(-1).unsqueeze(-1)

        # Calc target vals
        if targets is None:
            q_targets = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        else:
            q_targets = targets
        # Get expected Q values from local model
        q_expected, taus = self._get_q_preds(obs, actions)
        # Quantile Huber loss
        loss = self._quantile_loss(q_expected, q_targets, taus, batch_size)
        loss = loss.mean()
        return loss

    def _quantile_loss(self, preds, targets, taus, batch_size):
        td_error = targets - preds
        assert td_error.shape == (batch_size, self.nets[0].num_tau, self.nets[0].num_tau), \
            f'Wrong td error shape: {td_error.shape}. target: {targets.shape}. expected: {preds.shape}'
        huber_l = calculate_huber_loss(td_error, self.huber_thresh)
        quantil_l = (taus - (td_error.detach() < 0).float()).abs() * huber_l / self.huber_thresh
        # loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights # FOR PER!
        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        return loss

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        batch_size = obs.shape[0]

        cos, taus = self.target_nets[0].sample_cos(batch_size)
        num_taus = taus.shape[1]

        q_targets_next = self.next_obs_val(next_obs, cos, taus)
        gammas = self._calc_gammas(done_flags, extra_info).unsqueeze(-1).unsqueeze(-1)
        # Compute Q targets for current states
        q_targets = rewards.unsqueeze(-1) + (gammas * q_targets_next)
        assert q_targets.shape == (batch_size, 1, num_taus), \
            f"Wrong target shape: {q_targets.shape}"
        return q_targets

    def _get_q_preds(self, obs, actions):
        batch_size = obs.shape[0]
        cos, taus = self.nets[0].sample_cos(batch_size)
        q_k = self.obs_val(obs, cos, taus)
        num_taus = taus.shape[1]
        action_index = actions.unsqueeze(-1).expand(batch_size, num_taus, 1)
        q_expected = q_k.gather(2, action_index)
        assert q_expected.shape == (batch_size, num_taus, 1), f"Wrong shape: {q_expected.shape}"
        return q_expected, taus

    def next_obs_val(self, *args):
        """Calculate the value of the next obs via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        # Next state action selection
        max_idcs, q_vals_next = self.next_obs_act_select(*args)
        # Next state action evaluation
        q_vals_next = self.next_obs_act_eval(max_idcs, *args, q_vals_next_eval=q_vals_next)
        return q_vals_next

    def next_obs_act_select(self, next_obs, cos, taus, use_target_net=True):
        # Next state action selection
        batch_size = next_obs.shape[0]
        num_taus = taus.shape[1]
        q_quants_next = self.q_pred_next_state(next_obs, cos, taus, use_target_net=use_target_net)
        exp_q_next = q_quants_next.mean(dim=1)
        # Get max predicted Q values (for next states) from target model
        max_idcs = torch.argmax(exp_q_next, dim=1, keepdim=True)
        # Bring in same shape as q_targets_next:
        max_idcs = max_idcs.unsqueeze(-1).expand(batch_size, num_taus, 1)
        return max_idcs, q_quants_next

    def next_obs_act_eval(self, max_idcs, next_obs, cos, taus, q_vals_next_eval=None, use_target_net=True):
        if q_vals_next_eval is None:
            q_vals_next_eval = self.q_pred_next_state(next_obs, cos, taus, use_target_net=use_target_net)
        # Take max actions
        q_vals_next = q_vals_next_eval.gather(dim=2, index=max_idcs).transpose(1, 2)
        return q_vals_next

    def obs_val(self, obs, cos, taus, net=None):
        if net is None:
            net = self.q_net
        quants, _ = net.get_quantiles(obs, cos=cos, taus=taus)
        return quants

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, cos, taus, use_target_net=True, net=None):
        """ Specified in extra method to be potentially overridden by subclasses"""
        if net is None:
            if use_target_net:
                net = self.q_net_target
            else:
                net = self.q_net
        q_pred, _ = net.get_quantiles(next_obs, cos=cos, taus=taus)
        return q_pred


class V(Policy):
    """A state value network"""

    def __init__(self, obs_size, net: DictConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v_out_size = 1
        net = dict(net)
        del net["dueling"]
        print(obs_size, net)
        self.v_net = MLP(obs_size, v_out_size, **net)
        self.v_net_target = MLP(obs_size, v_out_size, **net)
        freeze_params(self.v_net_target)
        # Create net lists to update target nets
        self.nets = [self.v_net]
        self.target_nets = [self.v_net_target]

    def forward(self, obs):
        return self.v_net(obs)

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        preds = self.v_net(obs).squeeze()
        if targets is None:
            targets = self.calc_target_val(rewards, done_flags, next_obs)
        assert targets.shape == preds.shape, f'{preds.shape} - {targets.shape}'
        loss = (targets - preds) ** 2
        return loss.mean()

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        v_vals_next = self.next_obs_val(next_obs)
        gammas = self._calc_gammas(done_flags, extra_info)
        targets = rewards + gammas * v_vals_next
        return targets

    @torch.no_grad()
    def next_obs_val(self, next_obs):
        v_vals_next = self.v_net_target(next_obs)
        return v_vals_next.squeeze()
