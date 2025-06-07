# noqa: N806


import random

import torch

from roboro.base_policies import MultiNetPolicy, Q, V
from roboro.networks import IQNNet
from roboro.utils import (
    calculate_huber_loss,
    create_wrapper,
    freeze_params,
    unsqueeze_to,
)


def create_q(
    *q_args,
    double_q: bool = False,
    soft_q: bool = False,
    munch_q: bool = False,
    iqn: bool = False,
    int_ens: bool = False,
    rem: bool = False,
    **policy_kwargs,
) -> Q:
    PolicyClass = Q  # noqa: N806

    if double_q:
        PolicyClass = create_wrapper(DoubleQ, PolicyClass)  # noqa: N806
    if iqn:
        PolicyClass = create_wrapper(IQN, PolicyClass)  # noqa: N806
    if soft_q or munch_q:
        PolicyClass = create_wrapper(SoftQ, PolicyClass)  # noqa: N806
        if munch_q:
            PolicyClass = create_wrapper(MunchQ, PolicyClass)  # noqa: N806
    if int_ens:
        PolicyClass = create_wrapper(InternalEnsemble, PolicyClass)  # noqa: N806
    elif rem:
        PolicyClass = create_wrapper(REM, PolicyClass)  # noqa: N806
    q = PolicyClass(*q_args, **policy_kwargs)

    return q


def create_v(
    *v_args,
    iqn: bool = False,
    int_ens: bool = False,
    rem: bool = False,
    **policy_kwargs,
) -> V:
    PolicyClass = V  # noqa: N806
    if iqn:
        PolicyClass = create_wrapper(IQN, PolicyClass)  # noqa: N806
    if int_ens:
        PolicyClass = create_wrapper(InternalEnsemble, PolicyClass)  # noqa: N806
    elif rem:
        PolicyClass = create_wrapper(REM, PolicyClass)  # noqa: N806
    v = PolicyClass(*v_args, **policy_kwargs)
    return v


def create_policy(
    obs_size,
    act_size,
    policy_kwargs,
    double_q=False,
    use_qv=False,
    use_qvmax=False,
    iqn=False,
    use_soft_q=False,
    use_munch_q=False,
    rem=False,
    int_ens=False,
):
    if int_ens:  # if user wants an ensemble
        # The q_creation_kwargs are for the base Q-networks inside the ensemble
        q_creation_kwargs = {
            "double_q": double_q,
            "iqn": iqn,
            "soft_q": use_soft_q,
            "munch_q": use_munch_q,
            "rem": False,  # REM is a different ensemble strategy, disable it
        }
        # policy_kwargs may contain ensemble specific params like size
        ensemble_kwargs = policy_kwargs.pop("ensemble", {})
        q_args = (obs_size, act_size)

        return EnsembleQ(q_args, q_creation_kwargs, policy_kwargs, **ensemble_kwargs)

    q_args = (obs_size, act_size)
    q_creation_kwargs = {
        "double_q": double_q,
        "iqn": iqn,
        "soft_q": use_soft_q,
        "munch_q": use_munch_q,
        "rem": rem,
        "int_ens": int_ens,
    }
    if use_qv or use_qvmax:
        v_args = (obs_size, act_size)
        v_creation_kwargs = {"iqn": iqn, "int_ens": int_ens, "rem": rem}
        if use_qv:
            policy = QV(
                q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs
            )
        else:
            policy = QVMax(
                q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs
            )
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
        # kwargs will include discretization_method, action_dims, num_bins_per_dim if independent_dims
        super().__init__(*args, **kwargs)

    def get_net_args(self):
        args = super().get_net_args()
        args = args + [self.num_tau, self.num_policy_samples]
        return args

    def __str__(self):
        return f"IQN <{super().__str__()}>"

    def create_net(self, target=False):
        net = IQNNet(*self.net_args, **self.net_kwargs)
        if target:
            freeze_params(net)
        return net

    def calc_loss(
        self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None
    ):
        # Reshape:
        batch_size = obs.shape[0]
        actions = actions.unsqueeze(-1)

        # Calc target vals
        targets = self.calc_target_val(
            obs, actions, rewards, done_flags, next_obs, next_obs_val=next_obs_val
        )
        # Get expected Q values from local model
        q_expected, taus = self._get_obs_preds(obs, actions)
        # Quantile Huber loss
        loss, tde = self._quantile_loss(q_expected, targets, taus, batch_size)
        if "sample_weight" in extra_info:
            loss *= extra_info["sample_weight"]
        return loss.mean(), abs(tde)

    def _quantile_loss(self, preds, targets, taus, batch_size):
        td_error = targets - preds  # preds: (B, N_tau_prime), targets: (B, 1, N_tau)
        # Ensure shapes are compatible for broadcasting. IQN paper uses (B, N_tau_prime, N_tau)
        # Here, preds are (B, num_tau_k) and targets are (B, 1, num_tau_k_prime)
        # If num_tau (from self.num_tau for preds) and num_tau_prime (from target net for targets) are same,
        # td_error will be (B, num_tau, num_tau) after broadcasting targets and preds. This is fine.
        # If they differ, one must be 1. Here target is (B,1,N') and pred is (B,N,1) (conceptually for broadcast)
        # Results in td_error (B, N, N')

        # Original code in project had: (batch_size, self.nets[0].num_tau, self.nets[0].num_tau)
        # This implies N_tau_prime and N_tau were identical (self.num_tau for online, self.num_tau for target via sample_cos)
        # Let's assume num_tau for online (preds) and target (targets) are the same (self.num_tau)
        # So preds: (B, num_tau), targets: (B, 1, num_tau) after Q.calc_target_val calls next_obs_val.
        # next_obs_val from IQN returns (B, num_tau), so targets from Q.calc_target_val will be (B, num_tau) after gamma etc.
        # We need to unsqueeze targets for broadcasting with taus in quantile_loss.
        # preds is (B, num_tau), targets is (B, num_tau)
        # td_error = targets.unsqueeze(1) - preds.unsqueeze(2) # (B, 1, N) - (B, N, 1) -> (B, N, N)
        # The current Q.calc_target_val returns targets of shape (B, num_tau) for IQN.
        # preds from self._get_obs_preds will be (B, num_tau).
        # So td_error = targets.unsqueeze(1) - preds.unsqueeze(2) is appropriate.

        # Let's re-verify shapes from call site:
        # q_expected from _get_obs_preds is (B, num_tau)
        # targets from calc_target_val (IQN version) is (B,1,num_tau)
        # So td_error = targets - q_expected.unsqueeze(1) # (B,1,N) - (B,N,1) gives (B,N,N) if q_exp is (B,N)
        # This is what the assert check was: (B, N_tau_online, N_tau_target)
        # For simplicity and current structure, let's assume targets is (B, N_tau_target) from calc_target_val
        # and q_expected (preds) is (B, N_tau_online) from _get_obs_preds
        # td_error: (B, N_tau_target, 1) - (B, 1, N_tau_online)
        td_error = targets.unsqueeze(2) - preds.unsqueeze(1)

        assert (
            td_error.shape[:1] == (batch_size,)
            and td_error.shape[1:]
            == (
                self.target_nets[0].num_tau,
                self.nets[0].num_tau,
            )
        ), f"Wrong td error shape: {td_error.shape}. Expected ({batch_size}, {self.target_nets[0].num_tau}, {self.nets[0].num_tau})"

        huber_l = calculate_huber_loss(td_error, self.huber_thresh)
        # taus is (B, N_tau_online, 1)
        # td_error.detach() < 0 is (B, N_tau_target, N_tau_online)
        # Need to align taus with the N_tau_online dimension (dim 2 of td_error)
        quantil_l = (
            (taus.unsqueeze(1) - (td_error.detach() < 0).float()).abs()
            * huber_l
            / self.huber_thresh
            # taus is (B, N_tau_online, 1) -> (B, 1, N_tau_online, 1)
            # (td_error <0) is (B, N_tau_target, N_tau_online)
            # This part needs careful check of original IQN paper for broadcasting rules if N_tau_online != N_tau_target
            # Assuming N_tau_online (from self.nets[0].num_tau or self.num_tau) and N_tau_target (from self.target_nets[0].num_tau or self.num_tau)
            # are the same for this implementation, matching the assert.
            # So taus is (B, N, 1). td_error is (B,N,N). (td < 0) is (B,N,N)
            # (taus - (td<0).float()).abs() should be: (B, N_target, N_online)
            # taus needs to be (B, 1, N_online) for broadcasting. (B, N_tau,1) from sample_cos
            # Original: (taus - (td_error.detach() < 0).float()).abs()
            # If taus is (B, N_tau_online, 1) and (td < 0) is (B, N_tau_target, N_tau_online)
            # then taus needs to be broadcast. (B, 1, N_tau_online) for (td<0)
            # ( (B,1,N_online) - (B,N_target,N_online) ).abs() -> (B, N_target, N_online)
        )
        # sum over N_tau_online (dim 2), mean over N_tau_target (dim 1)
        loss = quantil_l.sum(dim=2).mean(dim=1)
        # For PER tde, use mean over target taus, sum over online taus
        tde_for_per = td_error.mean(dim=1).sum(
            dim=1
        )  # Mean over N_target, sum over N_online
        return loss, tde_for_per

    @torch.no_grad()
    def calc_target_val(
        self, obs, actions, rewards, done_flags, next_obs, next_obs_val=None
    ):
        batch_size = obs.shape[0]

        if next_obs_val is None:
            next_obs_val = self.next_obs_val(next_obs)
        gammas = self._calc_gammas(done_flags)
        # Compute Q targets for current states
        rewards = unsqueeze_to(rewards, next_obs_val)
        gammas = unsqueeze_to(gammas, next_obs_val)
        q_targets = rewards + (gammas * next_obs_val)
        assert q_targets.shape == (
            batch_size,
            1,
            self.num_tau,
        ), f"Wrong target shape: {q_targets.shape}"
        return q_targets

    def next_obs_val(self, next_obs, *args, **kwargs):
        batch_size = next_obs.shape[0]
        cos, taus = self.target_nets[0].sample_cos(batch_size)
        return super().next_obs_val(next_obs, cos, taus, *args, **kwargs)

    def _get_obs_preds(self, obs, actions):
        batch_size = obs.shape[0]
        # cos and taus from online network's perspective (self.num_tau)
        cos, taus = self.nets[0].sample_cos(batch_size, num_tau=self.num_tau)
        # quants_flat has shape (batch_size, self.num_tau, self.act_size)
        # where self.act_size is policy_act_shape (action_dims * num_bins_per_dim for independent)
        quants_flat = self.obs_val(obs, cos, taus)

        if self.discretization_method == "independent_dims":
            # actions is (batch, action_dims) - indices of chosen bins
            # Reshape quants_flat to (batch, num_tau, action_dims, num_bins_per_dim)
            quants_structured = quants_flat.view(
                batch_size, self.num_tau, self.action_dims, self.num_bins_per_dim
            )
            # Expand actions for gathering: (batch, action_dims) -> (B, 1, action_dims, 1)
            # So it can gather from (B, num_tau, action_dims, num_bins)
            actions_expanded = (
                actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_tau, -1, 1)
            )
            # Gather quantiles for the chosen bin in each dimension, for each tau sample
            # Result shape: (batch, num_tau, action_dims)
            chosen_bin_quants_per_dim = quants_structured.gather(
                3,
                actions_expanded.long(),  # Gather along num_bins_per_dim dimension (dim 3)
            ).squeeze(-1)
            # Sum quantiles across dimensions for the final Q-quantiles of the composite action
            # Result shape: (batch, num_tau)
            final_quants_pred = chosen_bin_quants_per_dim.sum(dim=2)
            chosen_action_q_vals = final_quants_pred
        else:  # joint discretization
            # actions is (batch,) or (batch, 1) - single discrete action index
            # quants_flat is (batch, num_tau, total_discrete_actions)
            # Need to gather along the last dimension (total_discrete_actions)
            # actions shape for gather: (batch, num_tau, 1)
            actions_expanded = actions.view(-1, 1, 1).expand(-1, self.num_tau, 1)
            chosen_action_q_vals = quants_flat.gather(
                dim=-1, index=actions_expanded.long()
            ).squeeze(-1)
            # Result shape: (batch, num_tau)

        return chosen_action_q_vals, taus  # taus is (batch_size, self.num_tau, 1)

    def next_obs_act_select(self, next_obs, *args, use_target_net=True, **kwargs):
        # cos and taus are passed in *args from the Q.next_obs_val -> IQN.next_obs_val call chain
        # These cos, taus are for the target network's num_tau (self.target_nets[0].num_tau)
        # which should be self.num_tau if target net created with same num_tau.
        # q_pred_next_state will use these to get quantiles from the appropriate net (online/target)
        q_quants_next_flat = self.q_pred_next_state(
            next_obs, *args, use_target_net=use_target_net, **kwargs
        )  # Shape: (B, num_target_tau, policy_act_shape)

        if self.discretization_method == "independent_dims":
            # q_quants_next_flat is (B, num_target_tau, action_dims * num_bins)
            q_quants_next_structured = q_quants_next_flat.view(
                next_obs.shape[0],
                -1,
                self.action_dims,
                self.num_bins_per_dim,  # -1 for num_target_tau
            )
            # Mean over tau dimension to get expected Q-values for each bin
            # Shape: (B, action_dims, num_bins_per_dim)
            exp_q_next_structured = q_quants_next_structured.mean(dim=1)
            # Get max predicted Q values (for next states) from target model
            max_idcs = torch.argmax(
                exp_q_next_structured, dim=2
            )  # (batch, action_dims)
            # Pass the original flat quantiles as context
            return max_idcs, q_quants_next_flat
        else:  # joint
            # q_quants_next_flat is (B, num_target_tau, total_discrete_actions)
            # Mean over tau dimension to get expected Q-values
            exp_q_next = q_quants_next_flat.mean(dim=1)  # (B, total_discrete_actions)
            max_idcs = torch.argmax(exp_q_next, dim=1, keepdim=True)  # (B, 1)
            return max_idcs, q_quants_next_flat

    def obs_val(self, obs, cos, taus, net=None):
        if net is None:
            net = self.q_net
        quants, _ = net.get_quantiles(obs, cos=cos, taus=taus)
        return quants

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, cos, taus, use_target_net=True, net=None):
        """Specified in extra method to be potentially overridden by subclasses"""
        if net is None:
            if use_target_net:
                net = self.q_net_target
            else:
                net = self.q_net
        q_pred, _ = net.get_quantiles(next_obs, cos=cos, taus=taus)
        return q_pred


class DoubleQ(Q):
    def __str__(self):
        return f"Double<{super().__str__()}>"

    def next_obs_val(self, *args, **kwargs):
        """Calculate the value of the next obs according to the double Q learning rule.
        It decouples the action selection (done via online network) and the action evaluation (done via target network).
        """
        # Next state action selection
        max_idcs, _ = self.next_obs_act_select(*args, use_target_net=False, **kwargs)
        # Next state action evaluation
        q_vals_next = self.next_obs_act_eval(
            max_idcs, *args, use_target_net=True, **kwargs
        )
        return q_vals_next


class InternalEnsemble(Q):
    def __init__(self, *args, size=4, **kwargs):
        """Implements an internal ensemble that averages over the prediction of many Q-heads."""
        # For reproducible ensemble initialization, create different seeds for each member
        # while maintaining overall reproducibility
        self.size = size

        # Store original RNG state
        original_torch_state = torch.get_rng_state()

        # Create ensemble with different but deterministic initialization
        super().__init__(*args, **kwargs)
        self.q_net = None
        self.q_net_target = None

        # Create nets with different seeds for diversity
        torch.manual_seed(torch.initial_seed() + 1000)  # Offset for ensemble
        self.nets = torch.nn.ModuleList()
        self.target_nets = torch.nn.ModuleList()

        for i in range(size):
            # Set different seed for each ensemble member
            torch.manual_seed(torch.initial_seed() + 1000 + i)
            self.nets.append(self.create_net())
            torch.manual_seed(torch.initial_seed() + 1000 + i)  # Same seed for target
            self.target_nets.append(self.create_net(target=True))

        # Restore original RNG state
        torch.set_rng_state(original_torch_state)

    def __str__(self):
        return f"IntEns_{self.size}<{super().__str__()}>"

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
        # print(args, kwargs)
        preds = torch.stack(
            [pred_next(next_obs, *args, net=net, **kwargs) for net in nets]
        )
        pred = self.agg_preds(preds)
        return pred

    def obs_val(self, obs, *args, **kwargs):
        obs_func = super().obs_val
        # print(obs_func(obs, *args, net=self.nets[0], **kwargs))
        preds = torch.stack(
            [obs_func(obs, *args, net=net, **kwargs) for net in self.nets]
        )
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
        return f"REM_{self.size}<{super().__str__()}>"

    def calc_loss(self, obs, *args, **kwargs):
        if self.alphas is None:
            self.alphas = self.gen_alphas(obs)
        loss = super().calc_loss(obs, *args, **kwargs)
        self.alphas = None
        return loss

    def obs_val(self, obs, *args, **kwargs):
        if self.alphas is None:
            self.alphas = self.gen_alphas(obs)
        return super().obs_val(obs, *args, **kwargs)

    def next_obs_val(self, next_obs, *args, **kwargs):
        if self.alphas is None:
            self.alphas = self.gen_alphas(next_obs)
        return super().next_obs_val(next_obs, *args, **kwargs)

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
        return f"Soft_{self.tau}_{self.l0}<{super().__str__()}>"

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, *args, use_target_net=True, **kwargs):
        next_state_q_vals = super().q_pred_next_state(
            next_obs, *args, use_target_net=use_target_net, **kwargs
        )
        next_state_policy_distr = torch.softmax(next_state_q_vals / self.tau, dim=-1)
        next_state_preds = next_state_q_vals - self._calc_entropy(next_state_q_vals)
        return (next_state_policy_distr * next_state_preds).sum(dim=-1).unsqueeze(-1)

    def _calc_entropy(self, q_vals):
        return torch.clamp(
            self.tau * torch.log_softmax(q_vals / self.tau, dim=-1), min=self.l0
        )


class MunchQ(SoftQ):
    def __init__(self, *args, alpha=0.9, **kwargs):
        """Munchausen Q-learning"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def __str__(self):
        return f"Munchausen_{self.alpha}<{super().__str__()}>"

    @torch.no_grad()
    def calc_target_val(self, obs, actions, *args, **kwargs):
        q_targets = super().calc_target_val(obs, actions, *args, **kwargs)
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

    def __init__(
        self, q_args, q_creation_kwargs, v_args, v_creation_kwargs, **policy_kwargs
    ):
        super().__init__()
        self.v = create_v(*v_args, **v_creation_kwargs, **policy_kwargs)
        self.q = create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
        self.policies = [self.v, self.q]

    def __str__(self):
        return f"QV<{str(self.q)}>"

    def forward(self, obs):
        return self.q(obs)

    def calc_loss(self, *loss_args):
        next_obs = loss_args[4]
        v_next_obs_val = self.v.next_obs_val(next_obs)
        loss, abs_tde = self._calc_qv_loss(
            *loss_args, q_next_obs_val=v_next_obs_val, v_next_obs_val=v_next_obs_val
        )
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
        return f"QVMax <{str(self.q)}>"

    def calc_loss(self, *loss_args):
        next_obs = loss_args[4]
        q_next_obs_val = self.q.next_obs_val(next_obs)
        v_next_obs_val = self.v.next_obs_val(next_obs)
        loss, abs_tde = self._calc_qv_loss(
            *loss_args, q_next_obs_val=v_next_obs_val, v_next_obs_val=q_next_obs_val
        )
        return loss, abs_tde


class Ensemble(MultiNetPolicy):
    def __init__(
        self,
        PolicyClass,
        size=1,
        *policy_args,
        **policy_kwargs,  # noqa: N803
    ):  # noqa: N803
        super().__init__()
        self.size = size
        self.policies = [
            PolicyClass(*policy_args, **policy_kwargs) for _ in range(size)
        ]
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


class EnsembleQ(MultiNetPolicy):
    def __init__(
        self,
        q_args: tuple,
        q_creation_kwargs: dict,
        policy_kwargs: dict,
        size: int = 5,
        num_sampled_nets: int = 2,
    ):
        super().__init__(gamma=policy_kwargs.get("gamma"))
        self.size = size
        self.num_sampled_nets = num_sampled_nets
        if self.num_sampled_nets > self.size:
            print(
                f"Warning: num_sampled_nets ({self.num_sampled_nets}) > ensemble size ({self.size})."
                f"Clamping to {self.size}."
            )
            self.num_sampled_nets = self.size

        self.policies: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                create_q(*q_args, **q_creation_kwargs, **policy_kwargs)
                for _ in range(size)
            ]
        )

        # for target net updates
        self.nets = []
        self.target_nets = []
        for p in self.policies:
            self.nets.extend(p.nets)
            self.target_nets.extend(p.target_nets)

    def __str__(self):
        return f"EnsembleQ(size={self.size}, sampled={self.num_sampled_nets})<{str(self.policies[0])}>"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        sampled_policies = random.sample(list(self.policies), self.num_sampled_nets)
        q_values = torch.stack([p(obs) for p in sampled_policies])
        min_q_values, _ = torch.min(q_values, dim=0)
        return min_q_values

    @torch.no_grad()
    def get_next_obs_val(self, next_obs: torch.Tensor) -> torch.Tensor:
        sampled_policies = random.sample(list(self.policies), self.num_sampled_nets)

        # Each policy uses its own target network(s) to calculate next_obs_val
        next_q_vals = torch.stack([p.next_obs_val(next_obs) for p in sampled_policies])
        min_next_q_vals, _ = torch.min(next_q_vals, dim=0)
        return min_next_q_vals

    def calc_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done_flags: torch.Tensor,
        next_obs: torch.Tensor,
        extra_info: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate shared next_obs_val using min of random subset of target policies
        next_obs_val = self.get_next_obs_val(next_obs)

        total_loss = torch.tensor(0.0, device=obs.device)
        all_tdes = []
        for policy in self.policies:
            # Each sub-policy calculates its loss using the shared target value
            loss, tde = policy.calc_loss(
                obs,
                actions,
                rewards,
                done_flags,
                next_obs,
                extra_info,
                next_obs_val=next_obs_val,
            )
            total_loss += loss
            all_tdes.append(tde)

        # For PER, we need a single TDE. We can average them.
        mean_abs_tde = torch.stack(all_tdes).mean(dim=0)

        return total_loss / self.size, mean_abs_tde
