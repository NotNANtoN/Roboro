import torch
from omegaconf import DictConfig

from roboro.networks import MLP, IQNNet


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


class Policy(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def _calc_gammas(self, extra_info):
        if "n_step" in extra_info:
            gammas = self.gamma ** extra_info["n_step"]
        else:
            gammas = self.gamma
        return gammas


class Q(Policy):
    def __init__(self, obs_shape, act_shape, gamma=0.99, double_q=False, net: DictConfig = None):
        super().__init__(gamma)
        self.gamma = gamma
        self.q_net = MLP(obs_shape, act_shape, **net)
        self.q_net_target = MLP(obs_shape, act_shape, **net)
        freeze_params(self.q_net_target)
        # Decide on using double Q-learning by choosing which method to use to calc next state value:
        if double_q:
            self._next_state_func = self._calc_next_obs_q_vals_double_q
        else:
            self._next_state_func = self._calc_next_obs_q_vals

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
        q_vals_next = self._next_state_func(next_obs, done_flags)
        assert q_vals_next.shape == rewards.shape
        gammas = self._calc_gammas(extra_info)
        targets = rewards + gammas * q_vals_next
        return targets

    def update_target_nets_hard(self):
        copy_weights(self.q_net, self.q_net_target)

    def update_target_nets_soft(self, val):
        polyak_update(self.q_net, self.q_net_target, val)

    @torch.no_grad()
    def _q_pred_next_state(self, next_obs, done_flags, net):
        """ Specified in function to be potentially overriden by subclasses"""
        return net(next_obs) * (~done_flags.unsqueeze(-1))

    def _calc_next_obs_q_vals(self, next_obs, done_flags):
        """Calculate the value of the next state via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        q_vals_next = self._q_pred_next_state(next_obs, done_flags, self.q_net_target)
        q_vals_next = torch.max(q_vals_next, dim=1)[0]
        return q_vals_next
        
    def _calc_next_obs_q_vals_double_q(self, next_obs, done_flags):
        """Calculate the value of the next state according to the double Q learning rule.
        It decouples the action selection (done via online network) and the action evaluation (done via target network).
        """
        # Next state action selection
        q_vals_next_online_net = self._q_pred_next_state(next_obs, done_flags, self.q_net)
        max_idcs = torch.max(q_vals_next_online_net, dim=1)[1]
        # Next state action evaluation
        q_vals_next_target_net = self._q_pred_next_state(next_obs, done_flags, self.q_net_target)
        q_vals_next = q_vals_next_target_net.gather(dim=1, index=max_idcs.unsqueeze(1)).squeeze()
        return q_vals_next

    def _get_q_preds(self, obs, actions):
        """Get Q-value predictions for current obs based on actions"""
        pred_q_vals = self.q_net(obs)
        # TODO: isn't here a better method than this stupid gather?
        chosen_action_q_vals = pred_q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        return chosen_action_q_vals


class SoftQ(Q):
    def __init__(self, *args, tau=0.03, l0=-1, **kwargs):
        """Entropy-regularized Q-learning. Clip entropy to l0 (=-1) to avoid numeric instability in case of a
        deterministic policy (would otherwise lead to -inf values)."""
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.l0 = l0

    @torch.no_grad()
    def _q_pred_next_state(self, next_obs, done_flags, net):
        next_state_q_vals = super()._q_pred_next_state(next_obs, done_flags, net)
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

    @torch.no_grad()
    def calc_target_val(self, obs, actions, rewards, done_flags, next_obs, extra_info):
        q_targets = super(MunchQ, self).calc_target_val(obs, actions, rewards, done_flags, next_obs, extra_info)
        munch_reward = self._calc_entropy(self.q_net_target(obs))
        munch_reward = munch_reward.gather(1, actions.unsqueeze(-1)).squeeze()
        munchausen_targets = q_targets + self.alpha * munch_reward
        return munchausen_targets


class V(Policy):
    """A state value network"""
    def __init__(self, obs_shape, gamma=0.99, net: DictConfig = None):
        super().__init__(gamma)
        v_out_size = 1
        net = dict(net)
        del net["dueling"]
        self.v_net = MLP(obs_shape, v_out_size, **net)
        self.v_net_target = MLP(obs_shape, v_out_size, **net)
        freeze_params(self.v_net_target)
        
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
        v_vals_next = self._calc_next_obs_vals(next_obs, done_flags)
        gammas = self._calc_gammas(extra_info)
        targets = rewards + gammas * v_vals_next
        return targets

    def update_target_nets_hard(self):
        copy_weights(self.v_net, self.v_net_target)

    def update_target_nets_soft(self, val):
        polyak_update(self.v_net, self.v_net_target, val)

    @torch.no_grad()
    def _calc_next_obs_vals(self, next_obs, done_flags):
        """If a done_flag is set the next obs val is 0, else calculate it"""
        v_vals_next = self.v_net_target(next_obs) * (~done_flags.unsqueeze(-1))
        return v_vals_next.squeeze()


class QV(torch.nn.Module):
    """Train a state-action value network (Q) network and an additional state value network (V) and
    train the Q net using the V net.
    """
    def __init__(self, obs_shape, act_shape, gamma=0.99, double_q=False, net: DictConfig = None):
        super().__init__()
        self.v = V(obs_shape, gamma, net=net)
        self.q = Q(obs_shape, act_shape, gamma, double_q, net=net)
        # TODO: if dueling is set, try to incorporate the V net into the Q net

    def forward(self, obs):
        return self.q(obs)

    def calc_loss(self, *loss_args):
        v_target = self.v.calc_target_val(*loss_args)
        loss = self._calc_qv_loss(*loss_args, q_target=v_target, v_target=v_target)
        return loss
        
    def update_target_nets_hard(self):
        self.v.update_target_nets_hard()
        self.q.update_target_nets_hard()

    def update_target_nets_soft(self, val):
        self.v.update_target_nets_soft(val)
        self.q.update_target_nets_soft(val)

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


class IQN(SoftQ):
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
        gammas = self._calc_gammas(extra_info)
        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            action_idx = torch.argmax(exp_q_next, dim=1, keepdim=True)
            # Bring in same shape as q_targets_next:
            action_idx = action_idx.unsqueeze(-1).expand(batch_size, self.num_quantiles, 1)
            # Take max actions
            q_targets_next = q_quants_next.gather(dim=2, index=action_idx).transpose(1, 2)
            # Compute Q targets for current states
            q_targets = rewards.unsqueeze(-1) + (gammas * q_targets_next * (~done_flags))
        else:
            # calculate log-pi
            tau_log_pi_next = self._calc_entropy(exp_q_next).unsqueeze(1)

            pi_target = torch.softmax(exp_q_next / self.tau, dim=1).unsqueeze(1)
            next_state_vals = (gammas *
                         (pi_target * (q_quants_next - tau_log_pi_next) * (~done_flags)).sum(2)
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


class Ensemble(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, PolicyClass):
        super().__init__(obs_shape)
        self.policies = [PolicyClass]

    def calc_loss(self, *args):
        losses = torch.stack([policy.calc_loss(*args) for policy in self.policies])
        loss = losses.mean()
        return loss
