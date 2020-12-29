import torch
from omegaconf import DictConfig

from roboro.networks import MLP, IQNNet
from roboro.utils import create_wrapper


def polyak_update(net, target_net, factor):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(factor * target_param.data + param.data * (1.0 - factor))


def copy_weights(source_net, target_net):
    target_net.load_state_dict(source_net.state_dict())


def freeze_params(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    # assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss


def create_q(*args, double_q=False, soft_q=False, munch_q=False, iqn=False, **policy_kwargs):
    PolicyClass = Q
    if iqn:
        PolicyClass = create_wrapper(IQNQ, PolicyClass)
        #policy_kwargs.update({})
    if double_q:
        PolicyClass = create_wrapper(DoubleQ, PolicyClass)
    elif soft_q or munch_q:
        PolicyClass = create_wrapper(SoftQ, PolicyClass)
        if munch_q:
            PolicyClass = create_wrapper(MunchQ, PolicyClass)
    policy = PolicyClass(*args, **policy_kwargs)
    return policy


def create_policy(obs_size, act_size, policy_kwargs,
                  double_q=False, use_qv=False, use_qvmax=False, iqn=False, use_soft_q=False, use_munch_q=False):
    if use_qv:
        policy = QV()
    elif use_qvmax:
        policy = QVMax()
    else:
        policy = create_q(obs_size, act_size, double_q=double_q, soft_q=use_soft_q, munch_q=use_munch_q, iqn=iqn,
                          **policy_kwargs)
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
        q_vals_next = self.next_state_val(next_obs)
        assert q_vals_next.shape == rewards.shape
        gammas = self._calc_gammas(done_flags, extra_info)
        targets = rewards + gammas * q_vals_next
        return targets

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, net):
        """ Specified in extra method to be potentially overridden by subclasses"""
        return net(next_obs)

    def next_state_val(self, next_obs):
        """Calculate the value of the next state via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = self.q_pred_next_state(next_obs, self.q_net_target)
        q_vals_next = torch.max(q_vals_next, dim=1)[0]
        return q_vals_next

    def _get_q_preds(self, obs, actions):
        """Get Q-value predictions for current obs based on actions"""
        pred_q_vals = self.q_net(obs)
        # TODO: isn't there a better method than this stupid gather?
        chosen_action_q_vals = pred_q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        return chosen_action_q_vals


class DoubleQ(Q):
    def __str__(self):
        return f'DoubleQ <{super().__str__()}>'

    def next_state_val(self, next_obs):
        """Calculate the value of the next state according to the double Q learning rule.
        It decouples the action selection (done via online network) and the action evaluation (done via target network).
        """
        # Next state action selection
        q_vals_next_online_net = self.q_pred_next_state(next_obs, self.q_net)
        max_idcs = torch.max(q_vals_next_online_net, dim=1)[1]
        # Next state action evaluation
        q_vals_next_target_net = self.q_pred_next_state(next_obs, self.q_net_target)
        q_vals_next = q_vals_next_target_net.gather(dim=1, index=max_idcs.unsqueeze(1)).squeeze()
        return q_vals_next


class SoftQ(Q):
    def __init__(self, *args, tau=0.03, l0=-1, **kwargs):
        """Entropy-regularized Q-learning. Clip entropy to l0 (=-1) to avoid numeric instability in case of a
        deterministic policy (would otherwise lead to -inf values)."""
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.l0 = l0

    def __str__(self):
        return f'SoftQ_{self.tau}_{self.l0} <{super().__str__()}>'

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, net):
        next_state_q_vals = super().q_pred_next_state(next_obs, net)
        next_state_policy_distr = torch.softmax(next_state_q_vals, dim=1)
        next_state_preds = next_state_q_vals - self._calc_entropy(next_state_q_vals)
        return (next_state_policy_distr * next_state_preds).mean(dim=1).unsqueeze(-1)

    def _calc_entropy(self, q_vals):
        return torch.clamp(self.tau * torch.log_softmax(q_vals / self.tau, dim=1), min=self.l0)


class MunchQ:
    def __init__(self, *args, alpha=0.9, **kwargs):
        """Munchausen Q-learning"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_targets = super(MunchQ, self).calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        munch_reward = self._calc_entropy(self.q_net_target(obs))
        munch_reward = munch_reward.gather(1, actions.unsqueeze(-1)).squeeze()
        munchausen_targets = q_targets + self.alpha * munch_reward
        return munchausen_targets


class V(Policy):
    """A state value network"""
    def __init__(self, obs_size, net: DictConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v_out_size = 1
        net = dict(net)
        del net["dueling"]
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
        v_vals_next = self._calc_next_obs_vals(next_obs)
        gammas = self._calc_gammas(done_flags, extra_info)
        targets = rewards + gammas * v_vals_next
        return targets

    @torch.no_grad()
    def _calc_next_obs_vals(self, next_obs):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        v_vals_next = self.v_net_target(next_obs)
        return v_vals_next.squeeze()


class QV(Policy):
    """Train a state-action value network (Q) network and an additional state value network (V) and
    train the Q net using the V net.
    """
    def __init__(self, obs_shape, act_shape, gamma=0.99, net: DictConfig = None):
        super().__init__()
        # TODO: instead of using V and Q here, we need to define a constructor for these networks such that SoftQ, IQN,
        #  Ensembles etc can be also used in QV-learning
        self.v = V(obs_shape, gamma, net=net)
        self.q = Q(obs_shape, act_shape, gamma, net=net)
        # TODO: if dueling is set, try to incorporate the V net into the Q net
        # Create net lists to update target nets
        self.nets = [self.q.q_net, self.v.v_net]
        self.target_nets = [self.q.q_net_target, self.v.v_net_target]

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
    def calc_loss(self, *loss_args):
        q_target = self.q.calc_target_val(*loss_args)
        v_target = self.v.calc_target_val(*loss_args)
        loss = self._calc_qv_loss(*loss_args, q_target=v_target, v_target=q_target)
        return loss


class IQNQ(SoftQ):
    """
    IQN Agent that uses the IQN Layer and calculates a loss. Adapted from https://github.com/BY571/IQN-and-Extensions
    """
    def __init__(self, state_size, action_size, gamma=0.99, munchausen=False, tau=0.03, num_quantiles=8,
                 l0=-1, alpha=0.9,
                 net: DictConfig = None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
        """
        super().__init__(state_size, action_size, tau=tau, l0=l0, gamma=gamma, net=net)
        self.state_size = state_size
        self.action_size = action_size
        self.munchausen = munchausen
        # IQN hyperparams
        self.num_quantiles = num_quantiles
        # Munchausen hyperparams
        self.alpha = alpha
        # Create IQN-Network
        self.q_net = IQNNet(state_size, action_size, num_quantiles, **net)
        self.q_net_target = IQNNet(state_size, action_size, num_quantiles, **net)
        freeze_params(self.q_net_target)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None):
        batch_size = obs.shape[0]
        rewards = rewards.unsqueeze(-1)
        actions = actions.unsqueeze(-1)
        done_flags = done_flags.unsqueeze(-1).unsqueeze(-1)
        # Calc target vals
        q_targets = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        # Get expected Q values from local model
        q_expected, taus = self._get_q_preds(obs, actions)
        # Quantile Huber loss
        td_error = q_targets - q_expected
        assert td_error.shape == (batch_size, self.num_quantiles, self.num_quantiles), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
        # loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights # FOR PER!
        loss = quantil_l.sum(dim=1).mean(dim=1)  # , keepdim=True if per weights get multipl
        loss = loss.mean()
        return loss

    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        batch_size = obs.shape[0]
        q_quants_next, _ = self.q_net_target.get_quantiles(next_obs, self.num_quantiles)
        exp_q_next = q_quants_next.mean(dim=1)
        gammas = self._calc_gammas(done_flags, extra_info).unsqueeze(-1).unsqueeze(-1)
        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            action_idx = torch.argmax(exp_q_next, dim=1, keepdim=True)
            # Bring in same shape as q_targets_next:
            action_idx = action_idx.unsqueeze(-1).expand(batch_size, self.num_quantiles, 1)
            # Take max actions
            q_targets_next = q_quants_next.gather(dim=2, index=action_idx).transpose(1, 2)
            # Compute Q targets for current states
            q_targets = rewards.unsqueeze(-1) + (gammas * q_targets_next)
        else:
            # calculate log-pi
            tau_log_pi_next = self._calc_entropy(exp_q_next).unsqueeze(1)

            pi_target = torch.softmax(exp_q_next / self.tau, dim=1).unsqueeze(1)
            next_state_vals = (gammas.squeeze(-1) *
                               (pi_target * (q_quants_next - tau_log_pi_next)).sum(2)
                               ).unsqueeze(1)
            assert next_state_vals.shape == (batch_size, 1, self.num_quantiles)

            q_k_target = self.q_net_target(obs).detach()
            tau_log_pi_k = self._calc_entropy(q_k_target)
            assert tau_log_pi_k.shape == (batch_size, self.action_size), "shape instead is {}".format(
                tau_log_pi_k.shape)
            munchausen_addon = tau_log_pi_k.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.l0)).unsqueeze(-1)
            assert munchausen_reward.shape == (batch_size, 1, 1), f"Wrong shape: {munchausen_reward.shape}"

            q_targets = munchausen_reward + next_state_vals
        return q_targets

    def _get_q_preds(self, obs, actions):
        batch_size = obs.shape[0]
        q_k, taus = self.q_net.get_quantiles(obs, self.num_quantiles)
        action_index = actions.unsqueeze(-1).expand(batch_size, self.num_quantiles, 1)
        q_expected = q_k.gather(2, action_index)
        assert q_expected.shape == (batch_size, self.num_quantiles, 1), f"Wrong shape: {q_expected.shape}"
        return q_expected, taus


class Ensemble(Policy):
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


class REM(Ensemble):
    # TODO: This REM Ensemble most likely needs to be wrapped around the Q or V networks if used in QV,
    #  instead of wrapped around QV
    def __init__(self, *args, **kwargs):
        """Implements the Random Ensemble Mixture (REM)."""
        super().__init__(*args, **kwargs)

    def calc_loss(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        bs = obs.shape[0]
        alphas = torch.rand(bs)
        alphas /= alphas.sum()

        # calc preds
        preds = torch.stack([pol(obs) for pol in self.policies])
        pred = (preds * alphas).sum(dim=0)
        # calc targets
        target = self.calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info, alphas)

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info, alphas):
        next_vals = torch.stack([pol.next_state_func(next_obs) for pol in self.policies])
        next_val = (next_vals * alphas).sum(dim=0)
        assert next_val.shape == rewards.shape
        gammas = self._calc_gammas(done_flags, extra_info)
        targets = rewards + gammas * next_val
        return targets





