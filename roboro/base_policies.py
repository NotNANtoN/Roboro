import torch
from omegaconf import DictConfig, open_dict

from roboro.networks import MLP
from roboro.utils import polyak_update, copy_weights, freeze_params, unsqueeze_to, calculate_huber_loss


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
        self.obs_size = obs_size
        self.act_size = act_size
        self.net_config = net
        self.net_args = self.get_net_args()
        self.net_kwargs = self.get_net_kwargs()
        self.q_net = self.create_net()
        self.q_net_target = self.create_net(target=True)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]
        self.loss_fnc = torch.nn.MSELoss(reduction='none')

    def get_net_args(self):
        args = [self.obs_size, self.act_size]
        return args

    def get_net_kwargs(self):
        kwargs = self.net_config
        return kwargs

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

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None):
        preds = self._get_obs_preds(obs, actions)
        targets = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info,
                                       next_obs_val=next_obs_val)
        assert targets.shape == preds.shape, f"{targets.shape}, {preds.shape}"
        tde = (targets - preds)
        loss = calculate_huber_loss(tde)
        # loss = self.loss_fnc(preds, targets)
        if "sample_weight" in extra_info:
            loss *= extra_info["sample_weight"]
        return loss.mean(), abs(tde)

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None):
        if next_obs_val is None:
            next_obs_val = self.next_obs_val(next_obs)
        assert next_obs_val.shape == rewards.shape
        gammas = self._calc_gammas(done_flags, extra_info)

        targets = rewards + gammas * next_obs_val
        return targets

    def obs_val(self, obs, net=None):
        if net is None:
            net = self.q_net
        return net(obs)

    def _get_obs_preds(self, obs, actions):
        """Get Q-value predictions for current obs based on actions"""
        pred_q_vals = self.obs_val(obs)
        chosen_action_q_vals = self._gather_obs(pred_q_vals, actions)
        return chosen_action_q_vals.squeeze()

    def _gather_obs(self, preds, actions):
        actions = unsqueeze_to(actions, preds)
        actions = actions.expand(*preds.shape[:-1], 1)
        return preds.gather(dim=-1, index=actions)

    def next_obs_val(self, next_obs, *args, **kwargs):
        """Calculate the value of the next obs via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        # Next state action selection
        max_idcs, q_vals_next = self.next_obs_act_select(next_obs, *args, **kwargs)
        # Next state action evaluation
        q_vals_next = self.next_obs_act_eval(max_idcs, next_obs, *args, q_vals_next_eval=q_vals_next, **kwargs)
        return q_vals_next

    def next_obs_act_select(self, next_obs, *args, use_target_net=True, **kwargs):
        # Next state action selection
        q_vals_next = self.q_pred_next_state(next_obs, *args, use_target_net=use_target_net, **kwargs)
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


class V(Q):
    """Implements a state-value network by creating a Q-network with only one action.
    Only the _gather_preds function needs to be overwritten"""

    def __init__(self, obs_size, act_size, net: DictConfig = None, *args, **kwargs):
        super(V, self).__init__(obs_size, act_size, *args, net=net, **kwargs)

    def __str__(self):
        return f'V <{super().__str__()}>'

    def _gather_obs(self, preds, actions):
        """Instead of gathering the prediction according to the actions, simply return the predictions"""
        return preds

    def get_net_args(self):
        args = super().get_net_args()
        args[1] = 1  # set act_size to 1
        return args

    def get_net_kwargs(self):
        kwargs = super().get_net_kwargs()
        with open_dict(kwargs):
            del kwargs['dueling']
        return kwargs
