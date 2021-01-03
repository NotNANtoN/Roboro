import torch

from roboro.utils import create_wrapper
from roboro.base_policies import Q, IQNQ, V, MultiNetPolicy


class TypeNoQ(type):
    def mro(cls):
        current_mro = super().mro()
        modded_mro = tuple(superclass for superclass in current_mro if superclass is not Q)
        return modded_mro


def create_q(*q_args, double_q=False, soft_q=False, munch_q=False, iqn=False, int_ens=False, rem=False,
             **policy_kwargs):
    add_superclass = type
    if iqn:
        add_superclass = TypeNoQ
        PolicyClass = IQNQ
    else:
        PolicyClass = Q

    if double_q:
        PolicyClass = create_wrapper(DoubleQ, PolicyClass, add_superclass=add_superclass)
    if soft_q or munch_q:
        PolicyClass = create_wrapper(SoftQ, PolicyClass, add_superclass=add_superclass)
        if munch_q:
            PolicyClass = create_wrapper(MunchQ, PolicyClass, add_superclass=add_superclass)
    if int_ens:
        PolicyClass = create_wrapper(InternalEnsemble, PolicyClass, add_superclass=add_superclass)
    elif rem:
        PolicyClass = create_wrapper(REM, PolicyClass, add_superclass=add_superclass)
    q = PolicyClass(*q_args, **policy_kwargs)

    return q


def create_v(*v_args, iqn=False, **policy_kwargs):
    PolicyClass = V
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
        alphas = self.alphas
        while alphas.ndim < preds.ndim:
            alphas = alphas.unsqueeze(-1)
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
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_targets = super(MunchQ, self).calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        q_vals = self.forward_target(obs)
        munch_reward = self._calc_entropy(q_vals)
        while actions.ndim < munch_reward.ndim:
            actions = actions.unsqueeze(-1)
        munch_reward = munch_reward.gather(1, actions).squeeze()
        while munch_reward.ndim != q_targets.ndim:
            munch_reward = munch_reward.unsqueeze(-1)
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
