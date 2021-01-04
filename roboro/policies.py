import torch

from roboro.utils import create_wrapper, freeze_params, calculate_huber_loss, unsqueeze_to
from roboro.base_policies import Q, V, MultiNetPolicy
from roboro.networks import IQNNet


def create_q(*q_args, double_q=False, soft_q=False, munch_q=False, iqn=False, int_ens=False, rem=False,
             **policy_kwargs):
    PolicyClass = Q

    if iqn:
        PolicyClass = create_wrapper(IQN, PolicyClass)
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
    q = PolicyClass(*q_args, **policy_kwargs)

    return q


def create_v(*v_args, iqn=False, int_ens=False, rem=False, **policy_kwargs):
    PolicyClass = V
    if iqn:
        PolicyClass = create_wrapper(IQN, PolicyClass)
    if int_ens:
        PolicyClass = create_wrapper(InternalEnsemble, PolicyClass)
    elif rem:
        PolicyClass = create_wrapper(REM, PolicyClass)
    v = PolicyClass(*v_args, **policy_kwargs)
    return v


def create_policy(obs_size, act_size, policy_kwargs,
                  double_q=False, use_qv=False, use_qvmax=False, iqn=False, use_soft_q=False, use_munch_q=False,
                  rem=False, int_ens=False):
    q_args = (obs_size, act_size)
    q_creation_kwargs = {'double_q': double_q, 'iqn': iqn, 'soft_q': use_soft_q, 'munch_q': use_munch_q, 'rem': rem,
                         'int_ens': int_ens}
    if use_qv or use_qvmax:
        v_args = (obs_size, act_size)
        v_creation_kwargs = {'iqn': iqn, 'int_ens': int_ens, 'rem': rem}
        if use_qv:
            policy = QV(q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs)
        else:
            policy = QVMax(q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs)
    else:
        policy = create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
    return policy


class IQN(Q):
    """
    IQN Policy that uses the IQN Layer and calculates a loss. Adapted from https://github.com/BY571/IQN-and-Extensions
    """
    def __init__(self, *args, num_tau=16, num_policy_samples=32, **kwargs):
        self.huber_thresh = 1.0
        self.num_tau = num_tau
        self.num_policy_samples = num_policy_samples
        super().__init__(*args, **kwargs)

    def get_net_args(self):
        args = super().get_net_args()
        args = args + [self.num_tau, self.num_policy_samples]
        return args

    def __str__(self):
        return f'IQN <{super().__str__()}>'

    def create_net(self, target=False):
        net = IQNNet(*self.net_args, **self.net_kwargs)
        if target:
            freeze_params(net)
        return net

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None):
        # Reshape:
        batch_size = obs.shape[0]
        actions = actions.unsqueeze(-1)

        # Calc target vals
        targets = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info,
                                       next_obs_val=next_obs_val)
        # Get expected Q values from local model
        q_expected, taus = self._get_obs_preds(obs, actions)
        # Quantile Huber loss
        loss, tde = self._quantile_loss(q_expected, targets, taus, batch_size)
        if "sample_weight" in extra_info:
            loss *= extra_info["sample_weight"]
        return loss.mean(), abs(tde)

    def _quantile_loss(self, preds, targets, taus, batch_size):
        td_error = targets - preds
        assert td_error.shape == (batch_size, self.nets[0].num_tau, self.nets[0].num_tau), \
            f'Wrong td error shape: {td_error.shape}. target: {targets.shape}. expected: {preds.shape}'
        huber_l = calculate_huber_loss(td_error, self.huber_thresh)
        quantil_l = (taus - (td_error.detach() < 0).float()).abs() * huber_l / self.huber_thresh
        # loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights # FOR PER!
        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        return loss, td_error.sum(dim=1).mean(dim=1)

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None):
        batch_size = obs.shape[0]

        cos, taus = self.target_nets[0].sample_cos(batch_size)
        num_taus = taus.shape[1]

        if next_obs_val is None:
            next_obs_val = self.next_obs_val(next_obs, cos, taus)
        gammas = self._calc_gammas(done_flags, extra_info)
        # Compute Q targets for current states
        rewards = unsqueeze_to(rewards, next_obs_val)
        gammas = unsqueeze_to(gammas, next_obs_val)
        q_targets = rewards + (gammas * next_obs_val)
        assert q_targets.shape == (batch_size, 1, num_taus), \
            f"Wrong target shape: {q_targets.shape}"
        return q_targets

    def _get_obs_preds(self, obs, actions):
        batch_size = obs.shape[0]
        cos, taus = self.nets[0].sample_cos(batch_size)
        q_k = self.obs_val(obs, cos, taus)

        chosen_action_q_vals = self._gather_obs(q_k, actions)
        return chosen_action_q_vals, taus

    def next_obs_act_select(self, next_obs, *args, use_target_net=True, **kwargs):
        # Next state action selection
        q_quants_next = self.q_pred_next_state(next_obs, *args, use_target_net=use_target_net, **kwargs)
        exp_q_next = q_quants_next.mean(dim=1)
        # Get max predicted Q values (for next states) from target model
        max_idcs = torch.argmax(exp_q_next, dim=1, keepdim=True)
        # Bring in simiar shape as q_quants_next (except last dim):
        max_idcs = unsqueeze_to(max_idcs, q_quants_next)
        max_idcs = max_idcs.expand(*q_quants_next.shape[:-1], 1)
        return max_idcs, q_quants_next

    def next_obs_act_eval(self, max_idcs, next_obs, *args, q_vals_next_eval=None, use_target_net=True, **kwargs):
        if q_vals_next_eval is None:
            q_vals_next_eval = self.q_pred_next_state(next_obs, *args, use_target_net=use_target_net, **kwargs)
        # Take max actions
        q_vals_next = q_vals_next_eval.gather(dim=-1, index=max_idcs).transpose(1, 2)
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


class DoubleQ(Q):
    def __str__(self):
        return f'Double <{super().__str__()}>'

    def next_obs_val(self, *args, **kwargs):
        """Calculate the value of the next obs according to the double Q learning rule.
        It decouples the action selection (done via online network) and the action evaluation (done via target network).
        """
        # Next state action selection
        max_idcs, _ = self.next_obs_act_select(*args, use_target_net=False, **kwargs)
        # Next state action evaluation
        q_vals_next = self.next_obs_act_eval(max_idcs, *args, use_target_net=True, **kwargs)
        return q_vals_next


class InternalEnsemble(Q):
    def __init__(self, *args, size=4, **kwargs):
        """Implements an internal ensemble that averages over the prediction of many Q-heads."""
        super().__init__(*args, **kwargs)
        self.size = size
        self.q_net = None
        self.q_net_target = None
        self.nets = torch.nn.ModuleList(self.create_net() for _ in range(size))
        self.target_nets = torch.nn.ModuleList(self.create_net(target=True) for _ in range(size))

    def __str__(self):
        return f'IntEns_{self.size} <{super().__str__()}>'

    def forward(self, obs):
        preds = torch.stack([pol(obs) for pol in self.nets])
        mean_pred = torch.mean(preds, dim=0)
        return mean_pred

    @torch.no_grad()
    def forward_target(self, obs):
        preds = torch.stack([pol(obs) for pol in self.target_nets])
        mean_pred = torch.mean(preds, dim=0)
        return mean_pred

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, *args, use_target_net=True, **kwargs):
        if use_target_net:
            nets = self.target_nets
        else:
            nets = self.nets
        pred_next = super().q_pred_next_state
        preds = torch.stack([pred_next(next_obs, *args, net=net, **kwargs) for net in nets])
        pred = self.agg_preds(preds)
        return pred

    def obs_val(self, obs, *args, **kwargs):
        obs_func = super().obs_val
        # print(obs_func(obs, *args, net=self.nets[0], **kwargs))
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

    def calc_loss(self, *args, **kwargs):
        obs = args[0]
        if self.alphas is None:
            self.alphas = self.gen_alphas(obs)
        loss = super().calc_loss(*args, **kwargs)
        self.alphas = None
        return loss

    def calc_target_val(self, *args, **kwargs):
        obs = args[0]
        if self.alphas is None:
            self.alphas = self.gen_alphas(obs)
        loss = super().calc_target_val(*args, **kwargs)
        return loss

    def gen_alphas(self, obs):
        alphas = torch.rand(self.size, device=obs.device, dtype=obs.dtype)
        alphas /= alphas.sum()
        return alphas

    def agg_preds(self, preds):
        alphas = self.alphas
        alphas = unsqueeze_to(alphas, preds)
        preds = preds * alphas
        preds = preds.sum(dim=0)
        return preds


class SoftQ(Q):
    def __init__(self, *args, tau=0.03, l0=-1, **kwargs):
        """Entropy-regularized Q-learning. Clip entropy to l0 (=-1) to avoid numeric instability in case of a
        deterministic policy (would otherwise lead to -inf values)."""
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.l0 = l0

    def __str__(self):
        return f'Soft_{self.tau}_{self.l0} <{super().__str__()}>'

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, *args, use_target_net=True, **kwargs):
        next_state_q_vals = super().q_pred_next_state(next_obs, *args, use_target_net=use_target_net, **kwargs)
        next_state_policy_distr = torch.softmax(next_state_q_vals / self.tau, dim=-1)
        next_state_preds = next_state_q_vals - self._calc_entropy(next_state_q_vals)
        return (next_state_policy_distr * next_state_preds).sum(dim=-1).unsqueeze(-1)

    def _calc_entropy(self, q_vals):
        return torch.clamp(self.tau * torch.log_softmax(q_vals / self.tau, dim=-1), min=self.l0)


class MunchQ(SoftQ):
    def __init__(self, *args, alpha=0.9, **kwargs):
        """Munchausen Q-learning"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def __str__(self):
        return f'Munchausen_{self.alpha} <{super().__str__()}>'

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info, *args, **kwargs):
        q_targets = super(MunchQ, self).calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info, *args,
                                                        **kwargs)
        q_vals = self.forward_target(obs)
        munch_reward = self._calc_entropy(q_vals)
        actions = unsqueeze_to(actions, munch_reward)
        munch_reward = munch_reward.gather(1, actions).squeeze()
        munch_reward = unsqueeze_to(munch_reward, q_targets)
        munch_reward = munch_reward.expand(*q_targets.shape)
        munchausen_targets = q_targets + self.alpha * munch_reward
        return munchausen_targets


class QV(MultiNetPolicy):
    """Train a state-action value network (Q) network and an additional state value network (V) and
    train the Q net using the V net.
    """
    def __init__(self, q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs):
        super().__init__()
        self.v = create_v(*v_args, **v_creation_kwargs, **policy_kwargs)
        self.q = create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
        self.policies = [self.v, self.q]

    def __str__(self):
        return f'QV <{str(self.q)}>'

    def forward(self, obs):
        return self.q(obs)

    def calc_loss(self, *loss_args):
        v_next_obs_val = self.v.calc_target_val(*loss_args)
        loss, abs_tde = self._calc_qv_loss(*loss_args, q_next_obs_val=v_next_obs_val, v_next_obs_val=v_next_obs_val)
        return loss, abs_tde

    def _calc_qv_loss(self, *loss_args, q_next_obs_val=None, v_next_obs_val=None):
        v_loss, v_abs_tde = self.v.calc_loss(*loss_args, next_obs_val=v_next_obs_val)
        q_loss, q_abs_tde = self.q.calc_loss(*loss_args, next_obs_val=q_next_obs_val)
        loss = (v_loss + q_loss).mean()
        abs_tde = v_abs_tde + q_abs_tde
        return loss, abs_tde


class QVMax(QV):
    """QVMax is an off-policy variant of QV. The V-net is trained by using the Q-net and vice versa."""
    def __str__(self):
        return f'QVMax <{str(self.q)}>'

    def calc_loss(self, *loss_args):
        q_next_obs_val = self.q.calc_target_val(*loss_args)
        v_next_obs_val = self.v.calc_target_val(*loss_args)
        loss, abs_tde = self._calc_qv_loss(*loss_args, q_next_obs_val=v_next_obs_val, v_next_obs_val=q_next_obs_val)
        return loss, abs_tde


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
