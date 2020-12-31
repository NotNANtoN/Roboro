import torch
from omegaconf import DictConfig

from roboro.networks import MLP, IQNNet
from roboro.utils import create_wrapper, polyak_update, copy_weights, freeze_params, calculate_huber_loss


def create_q(*q_args, double_q=False, soft_q=False, munch_q=False, iqn=False, int_ens=False, rem=False,
             **policy_kwargs):
    #PolicyClass = Q
    # TODO: the combination of IQNQ with SoftQ does not work yet - for some reason the base policy Q is used...
    #  look into removing all superclasses within the create_wrapper method. The __bases__ field is not sufficient...
    if iqn:
        PolicyClass = IQNQ  #create_wrapper(IQNQ, PolicyClass)
    else:
        PolicyClass = Q

    if double_q:
        PolicyClass = create_wrapper(DoubleQ, PolicyClass)
    if soft_q or munch_q:
        PolicyClass = create_wrapper(SoftQ, PolicyClass)
        if munch_q:
            PolicyClass = create_wrapper(MunchQ, PolicyClass)
    if int_ens:
        PolicyClass = create_wrapper(InternalEnsemble, PolicyClass)
    elif rem:
        PolicyClass = create_wrapper(REM, PolicyClass)
    print(PolicyClass)
    print(PolicyClass.__bases__)
    quit()
    q = PolicyClass(*q_args, **policy_kwargs)
    return q


def create_v(*v_args, iqn=False, **policy_kwargs):
    PolicyClass = V
    # TODO: create IQN_V class to allow implicit quantile networks for V. This allows IQN+QV
    v = PolicyClass(*v_args, **policy_kwargs)
    return v


def create_policy(obs_size, act_size, policy_kwargs,
                  double_q=False, use_qv=False, use_qvmax=False, iqn=False, use_soft_q=False, use_munch_q=False,
                  rem=False, int_ens=False):
    q_args = (obs_size, act_size)
    q_creation_kwargs = {'double_q': double_q, 'iqn': iqn, 'soft_q': use_soft_q, 'munch_q': use_munch_q, 'rem': rem,
                         'int_ens': int_ens}
    if use_qv or use_qvmax:
        v_args = (obs_size,)
        v_creation_kwargs = {}
        if use_qv:
            policy = QV(q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs)
        else:
            policy = QVMax(q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs)
    else:
        policy = create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
    return policy


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
        gammas = (~done_flags).float()
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
        self.q_net = MLP(obs_size, act_size, **net)
        self.q_net_target = MLP(obs_size, act_size, **net)
        freeze_params(self.q_net_target)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

    def __str__(self):
        return f'Q <{super().__str__()}>'

    def forward(self, obs):
        return self.q_net(obs)

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

    def next_obs_val(self, next_obs):
        """Calculate the value of the next obs via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = self.q_pred_next_state(next_obs, use_target_net=True)
        q_vals_next = torch.max(q_vals_next, dim=1)[0]
        return q_vals_next

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, use_target_net=True):
        """ Specified in extra method to be potentially overridden by subclasses"""
        if use_target_net:
            net = self.q_net_target
        else:
            net = self.q_net
        return net(next_obs)


class IQNQ(Policy):
    """
    IQN Agent that uses the IQN Layer and calculates a loss. Adapted from https://github.com/BY571/IQN-and-Extensions
    """
    def __init__(self, obs_size, act_size, *args, munchausen=False, num_tau=8,
                 num_policy_samples=32,
                 net: DictConfig = None, **kwargs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            tau (float): tau for soft updating the network weights
            gamma (float): discount factor
        """
        super().__init__(*args, **kwargs) #state_size, action_size, tau=tau, l0=l0, gamma=gamma, net=net)
        self.act_size = act_size
        self.munchausen = munchausen
        self.huber_thresh = 1.0
        # Munchausen hyperparams
        #self.alpha = alpha
        # Create IQN-Network
        self.q_net = IQNNet(obs_size, act_size, num_tau, num_policy_samples, **net)
        self.q_net_target = IQNNet(obs_size, act_size, num_tau, num_policy_samples, **net)
        freeze_params(self.q_net_target)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

    def __str__(self):
        return f'IQN <{super().__str__()}>'

    def forward(self, obs):
        return self.q_net(obs)

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
        assert td_error.shape == (batch_size, self.q_net.num_tau, self.q_net.num_tau), \
            f'Wrong td error shape: {td_error.shape}. target: {targets.shape}. expected: {preds.shape}'
        huber_l = calculate_huber_loss(td_error, self.huber_thresh)
        quantil_l = (taus - (td_error.detach() < 0).float()).abs() * huber_l / self.huber_thresh
        # loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights # FOR PER!
        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        return loss

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        batch_size = obs.shape[0]

        q_quants_next, taus_target = self.q_net_target.get_quantiles(next_obs)

        num_taus = taus_target.shape[1]
        exp_q_next = q_quants_next.mean(dim=1)
        gammas = self._calc_gammas(done_flags, extra_info)

        # Get max predicted Q values (for next states) from target model
        action_idx = torch.argmax(exp_q_next, dim=1, keepdim=True)
        # Bring in same shape as q_targets_next:
        action_idx = action_idx.unsqueeze(-1).expand(batch_size, num_taus, 1)
        # Take max actions
        q_targets_next = q_quants_next.gather(dim=2, index=action_idx).transpose(1, 2)

        # Alternative implementation that also runs and learns. Takes max per quantile:
        #q_targets_next = q_quants_next.max(2)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards.unsqueeze(-1) + (gammas * q_targets_next)
        assert q_targets.shape == (batch_size, 1, num_taus), \
            f"Wrong target shape: {q_targets.shape}"

        if self.munchausen:
            # calculate log-pi
            entropy_next_obs = self._calc_entropy(exp_q_next).unsqueeze(1)
            pi_target = torch.softmax(exp_q_next / self.tau, dim=1).unsqueeze(1)

            next_obs_vals = (gammas.squeeze(-1) *
                             (pi_target * (q_quants_next - entropy_next_obs)).sum(2)
                             ).unsqueeze(1)
            assert next_obs_vals.shape == (batch_size, 1, self.num_quantiles)

            q_k_target = self.q_net_target(obs).detach()
            tau_log_pi_k = self._calc_entropy(q_k_target)
            assert tau_log_pi_k.shape == (batch_size, self.act_size), "shape instead is {}".format(
                tau_log_pi_k.shape)
            munchausen_addon = tau_log_pi_k.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.l0)).unsqueeze(-1)
            assert munchausen_reward.shape == (batch_size, 1, 1), f"Wrong shape: {munchausen_reward.shape}"

            q_targets = munchausen_reward + next_obs_vals
        return q_targets

    def _get_q_preds(self, obs, actions):
        batch_size = obs.shape[0]
        cos, taus = self.q_net.sample_cos(batch_size)
        q_k = self.obs_val(obs, cos=cos, taus=taus)
        num_taus = taus.shape[1]
        action_index = actions.unsqueeze(-1).expand(batch_size, num_taus, 1)
        q_expected = q_k.gather(2, action_index)
        assert q_expected.shape == (batch_size, num_taus, 1), f"Wrong shape: {q_expected.shape}"
        return q_expected, taus

    #def next_obs_val(self, next_obs):
    #    """Calculate the value of the next obs via the target network.
    #    If a done_flag is set the next obs val is 0, else calculate it"""
    #    q_vals_next = self.q_pred_next_state(next_obs, use_target_net=True)
    #    q_vals_next = torch.max(q_vals_next, dim=1)[0]
    #    return q_vals_next

    def obs_val(self, obs, cos, taus, net=None):
        if net is None:
            net = self.q_net
        quants, _ = net.get_quantiles(obs, cos=cos, taus=taus)
        return quants


class DoubleQ(Q):
    def __str__(self):
        return f'Double <{super().__str__()}>'

    def next_obs_val(self, next_obs):
        """Calculate the value of the next obs according to the double Q learning rule.
        It decouples the action selection (done via online network) and the action evaluation (done via target network).
        """
        # Next state action selection
        q_vals_next_online_net = self.q_pred_next_state(next_obs, use_target_net=False)
        max_idcs = torch.max(q_vals_next_online_net, dim=1)[1]
        # Next state action evaluation
        q_vals_next_target_net = self.q_pred_next_state(next_obs, use_target_net=True)
        q_vals_next = q_vals_next_target_net.gather(dim=1, index=max_idcs.unsqueeze(1)).squeeze()
        return q_vals_next


class InternalEnsemble(Q):
    def __init__(self, *args, size=4, **kwargs):
        """Implements an internal ensemble that averages over the prediction of many Q-heads."""
        super().__init__(*args, **kwargs)
        self.size = size
        obs_size, act_size = args
        net = kwargs['net']
        self.q_net = None
        self.q_net_target = None
        # TODO: the instantiation below does not allow IQN to be used in the ensemble
        self.nets = torch.nn.ModuleList([MLP(obs_size, act_size, **net) for _ in range(size)])
        self.target_nets = torch.nn.ModuleList([MLP(obs_size, act_size, **net) for _ in range(size)])
        for net in self.target_nets:
            freeze_params(net)

    def __str__(self):
        return f'IntEns_{self.size} <{super().__str__()}>'

    def forward(self, obs):
        preds = torch.stack([pol(obs) for pol in self.nets])
        mean_pred = torch.mean(preds, dim=0)
        return mean_pred

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, use_target_net=True):
        if use_target_net:
            nets = self.target_nets
        else:
            nets = self.nets
        preds = torch.stack([net(next_obs) for net in nets])
        pred = self.agg_preds(preds)
        return pred

    def obs_val(self, obs, *args, **kwargs):
        # TODO: wth is the line below necessary? if super().obs_val is used in the list comprehension it crashes...
        obs_func = super().obs_val
        #print(obs_func(obs, *args, net=self.nets[0], **kwargs))
        preds = torch.stack([obs_func(obs, *args, net=net, **kwargs) for net in self.nets])
        pred = self.agg_preds(preds)
        return pred

    def agg_preds(self, preds):
        return preds.mean(dim=0)


class REM(InternalEnsemble):
    def __init__(self, *args, **kwargs):
        """Implements the Random Ensemble Mixture (REM)."""
        super().__init__(*args, **kwargs)
        self.alphas = None

    def __str__(self):
        return f'REM_{self.size} <{super().__str__()}>'

    def calc_loss(self, *args):
        obs = args[0]
        self.alphas = self.gen_alphas(obs)
        loss = super().calc_loss(*args)
        self.alphas = None
        return loss

    def gen_alphas(self, obs):
        alphas = torch.rand(self.size, device=obs.device, dtype=obs.dtype)
        alphas /= alphas.sum()
        return alphas

    def agg_preds(self, preds):
        preds = preds * self.alphas.unsqueeze(-1).unsqueeze(-1)
        preds = preds.sum(dim=0)
        return preds

# TODO: I had to remove the inheritance from Q because it would always have Q as its superclass even if we want it
#  based on IQN. Maybe there is a way to filter out superclasses in our "create_wrapper" function such that we can keep
#  the inheritance here
class SoftQ:
    def __init__(self, *args, tau=0.03, l0=-1, **kwargs):
        """Entropy-regularized Q-learning. Clip entropy to l0 (=-1) to avoid numeric instability in case of a
        deterministic policy (would otherwise lead to -inf values)."""
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.l0 = l0

    def __str__(self):
        return f'Soft_{self.tau}_{self.l0} <{super().__str__()}>'

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, use_target_net=True):
        next_state_q_vals = super().q_pred_next_state(next_obs, use_target_net=use_target_net)
        next_state_policy_distr = torch.softmax(next_state_q_vals, dim=1)
        next_state_preds = next_state_q_vals - self._calc_entropy(next_state_q_vals)
        return (next_state_policy_distr * next_state_preds).mean(dim=1).unsqueeze(-1)

    def _calc_entropy(self, q_vals):
        return torch.clamp(self.tau * torch.log_softmax(q_vals / self.tau, dim=1), min=self.l0)


class MunchQ(SoftQ):
    def __init__(self, *args, alpha=0.9, **kwargs):
        """Munchausen Q-learning"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def __str__(self):
        return f'Munchausen_{self.alpha} <{super().__str__()}>'

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_targets = super(MunchQ, self).calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        munch_reward = self._calc_entropy(self.q_net_target(obs))
        if actions.ndim != munch_reward.ndim:
            actions = actions.unsqueeze(-1)
        munch_reward = munch_reward.gather(1, actions).squeeze()
        while munch_reward.ndim != q_targets.ndim:
            munch_reward = munch_reward.unsqueeze(-1)
        munch_reward = munch_reward.expand(*q_targets.shape)
        munchausen_targets = q_targets + self.alpha * munch_reward
        return munchausen_targets


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


class QV(MultiNetPolicy):
    """Train a state-action value network (Q) network and an additional state value network (V) and
    train the Q net using the V net.
    """
    def __init__(self, q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs):
        super().__init__()
        self.v = create_v(*v_args, **v_creation_kwargs, **policy_kwargs)
        self.q = create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
        # TODO: if dueling is set, try to incorporate the V net into the Q net
        self.policies = [self.v, self.q]

    def __str__(self):
        return f'QV <{str(self.q)}>'

    def forward(self, obs):
        return self.q(obs)

    def calc_loss(self, *loss_args):
        v_target = self.v.calc_target_val(*loss_args)
        loss = self._calc_qv_loss(*loss_args, q_target=v_target, v_target=v_target)
        return loss

    def _calc_qv_loss(self, *loss_args, q_target=None, v_target=None):
        v_loss = self.v.calc_loss(*loss_args, targets=v_target)
        q_loss = self.q.calc_loss(*loss_args, targets=q_target)
        loss = (v_loss + q_loss).mean()
        return loss


class QVMax(QV):
    """QVMax is an off-policy variant of QV. The V-net is trained by using the Q-net and vice versa."""
    def __str__(self):
        return f'QVMax <{str(self.q)}>'

    def calc_loss(self, *loss_args):
        q_target = self.q.calc_target_val(*loss_args)
        v_target = self.v.calc_target_val(*loss_args)
        loss = self._calc_qv_loss(*loss_args, q_target=v_target, v_target=q_target)
        return loss


class Ensemble(MultiNetPolicy):
    def __init__(self, PolicyClass, size=1, *policy_args, **policy_kwargs):
        super().__init__()
        self.size = size
        self.policies = [PolicyClass(*policy_args, **policy_kwargs) for _ in range(size)]
        self.nets = []
        self.target_nets = []
        for pol in self.policies:
            self.nets += pol.nets
            self.target_nets += pol.target_nets

    def forward(self, obs):
        preds = torch.stack([pol(obs) for pol in self.policies])
        mean_pred = torch.mean(preds, dim=0)
        return mean_pred

    def calc_loss(self, *args):
        losses = torch.stack([policy.calc_loss(*args) for policy in self.policies])
        loss = losses.mean()
        return loss
