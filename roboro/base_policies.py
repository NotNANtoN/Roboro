import torch
from omegaconf import DictConfig, open_dict

from roboro.networks import MLP
from roboro.utils import (
    calculate_huber_loss,
    copy_weights,
    freeze_params,
    polyak_update,
    unsqueeze_to,
)


class Policy(torch.nn.Module):
    def __init__(self, gamma=None, **kwargs):
        """Policy superclass that deals with basic and repetitiv tasks such as updating the target networks or
        calculating the gamma values.

        Subclasses need to define self.nets and self.target_nets such that the target nets are updated properly.
        """
        super().__init__()
        self.gamma = gamma
        self.nets = []
        self.target_nets = []

    def __str__(self):
        return f"Policy_{self.gamma}"

    def _calc_gammas(self, done_flags):
        """Apply the discount factor. If a done flag is set we discount by 0."""
        gammas = (~done_flags).float().squeeze() * self.gamma
        return gammas

    def update_target_nets_hard(self):
        for net, target_net in zip(self.nets, self.target_nets, strict=False):
            copy_weights(net, target_net)

    def update_target_nets_soft(self, val):
        for net, target_net in zip(self.nets, self.target_nets, strict=False):
            polyak_update(net, target_net, val)

    def forward(self, obs):
        raise NotImplementedError

    def calc_loss(
        self, obs, actions, rewards, done_flags, next_obs, extra_info, targets=None
    ):
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
        # act_size here is policy_act_shape from Agent
        # For independent_dims, it's action_dims * num_bins_per_dim
        # For joint, it's the total number of discrete actions
        self.act_size = act_size
        self.net_config = net

        # Get discretization_method and related params from kwargs if provided
        # These are passed down from Agent -> create_policy -> Q
        self.discretization_method = kwargs.get("discretization_method", "joint")
        if self.discretization_method == "independent_dims":
            self.action_dims = kwargs.get("action_dims")
            self.num_bins_per_dim = kwargs.get("num_bins_per_dim")
            if self.action_dims is None or self.num_bins_per_dim is None:
                raise ValueError(
                    "action_dims and num_bins_per_dim must be provided for independent_dims method"
                )
            assert self.act_size == self.action_dims * self.num_bins_per_dim
        else:
            self.action_dims = None
            self.num_bins_per_dim = None

        self.net_args = self.get_net_args()
        self.net_kwargs = self.get_net_kwargs()
        self.q_net = self.create_net()
        self.q_net_target = self.create_net(target=True)
        # Create net lists to update target nets
        self.nets = [self.q_net]
        self.target_nets = [self.q_net_target]

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
        return f"Q <{super().__str__()}>"

    def forward(self, obs):
        return self.q_net(obs)

    @torch.no_grad()
    def forward_target(self, obs):
        return self.q_net_target(obs)

    def calc_loss(
        self, obs, actions, rewards, done_flags, next_obs, extra_info, next_obs_val=None
    ):
        preds = self._get_obs_preds(obs, actions)
        targets = self.calc_target_val(
            obs, actions, rewards, done_flags, next_obs, next_obs_val=next_obs_val
        )
        assert targets.shape == preds.shape, f"{targets.shape}, {preds.shape}"
        tde = targets - preds
        loss = calculate_huber_loss(tde)
        if "sample_weight" in extra_info:
            loss *= extra_info["sample_weight"]
        return loss.mean(), abs(tde)

    @torch.no_grad()
    def calc_target_val(
        self, obs, actions, rewards, done_flags, next_obs, next_obs_val=None
    ):
        if next_obs_val is None:
            next_obs_val = self.next_obs_val(next_obs)
        assert next_obs_val.shape == rewards.shape
        gammas = self._calc_gammas(done_flags)
        assert gammas.shape == rewards.shape

        targets = rewards + gammas * next_obs_val
        return targets

    def obs_val(self, obs, net=None):
        if net is None:
            net = self.q_net
        return net(obs)

    def _get_obs_preds(self, obs, actions):
        """Get Q-value predictions for current obs based on actions"""
        pred_q_vals_flat = self.obs_val(obs)  # (batch, policy_act_shape)

        if self.discretization_method == "independent_dims":
            # actions is (batch, action_dims) - indices of chosen bins
            pred_q_vals_structured = pred_q_vals_flat.view(
                -1, self.action_dims, self.num_bins_per_dim
            )
            # Gather Q-values for the chosen bin in each dimension
            # actions.unsqueeze(-1) gives (batch, action_dims, 1)
            chosen_action_q_vals_per_dim = pred_q_vals_structured.gather(
                2, actions.unsqueeze(-1).long()
            ).squeeze(-1)  # -> (batch, action_dims)
            # Sum Q-values across dimensions for the final Q-value of the composite action
            chosen_action_q_vals = chosen_action_q_vals_per_dim.sum(
                dim=1
            )  # -> (batch,)
        else:  # joint discretization
            # actions is (batch,) or (batch, 1) - single discrete action index
            chosen_action_q_vals = self._gather_obs(pred_q_vals_flat, actions)
            chosen_action_q_vals = chosen_action_q_vals.squeeze()  # Ensure (batch,)

        return chosen_action_q_vals

    def _gather_obs(self, preds, actions):
        actions = unsqueeze_to(actions, preds)
        actions = actions.expand(*preds.shape[:-1], 1)
        return preds.gather(dim=-1, index=actions)

    def next_obs_val(self, next_obs, *args, **kwargs):
        """Calculate the value of the next obs via the target network.
        If a done_flag is set the next obs val is 0, else calculate it"""
        # Select best action for next state
        max_idcs, q_vals_next = self.next_obs_act_select(
            next_obs, *args, use_target_net=True, **kwargs
        )
        # Evaluate selected action for next state (possibly using a different network)
        q_vals_next = self.next_obs_act_eval(
            max_idcs, next_obs, *args, q_vals_next_eval=q_vals_next, **kwargs
        )
        return q_vals_next

    def next_obs_act_select(self, next_obs, *args, use_target_net=True, **kwargs):
        # Next state action selection
        q_vals_next_flat = self.q_pred_next_state(
            next_obs, *args, use_target_net=use_target_net, **kwargs
        )

        if self.discretization_method == "independent_dims":
            # q_vals_next_flat can be (B, policy_act_shape) or (B, N_tau, policy_act_shape) if IQN
            original_shape = q_vals_next_flat.shape
            batch_size = original_shape[0]
            num_leading_dims = (
                q_vals_next_flat.ndim - 1
            )  # All dims except the last one (policy_act_shape)

            if num_leading_dims == 1:  # (B, policy_act_shape)
                q_vals_next_structured = q_vals_next_flat.view(
                    batch_size, self.action_dims, self.num_bins_per_dim
                )
            elif num_leading_dims == 2:  # (B, N_tau, policy_act_shape)
                # This case is for IQN like policies
                num_tau_dim = original_shape[1]
                q_vals_next_structured = q_vals_next_flat.view(
                    batch_size, num_tau_dim, self.action_dims, self.num_bins_per_dim
                )
            else:
                raise ValueError(
                    f"Unexpected shape for q_vals_next_flat: {original_shape}"
                )

            # max_idcs are the chosen bin indices for each dimension
            # If N_tau is present, argmax should be over bins, keeping N_tau and action_dims
            # For IQN, next_obs_act_select in IQN class itself handles the mean over taus before argmax.
            # So, if this Q.next_obs_act_select is called by IQN's machinery, it might be that
            # q_vals_next_flat is already (B, action_dims * num_bins_per_dim) after IQN averages over taus.
            # However, IQN.next_obs_act_select directly returns max_idcs and q_quants_next_flat.
            # This Q.next_obs_act_select is primarily for non-IQN Q or if DoubleQ needs online net selection.
            # For DoubleQ with IQN, Agent calls policy.forward() which is IQN.forward(), which averages over taus.
            # Let's assume if called directly, and it's IQN, it's after mean. If not, then it's (B, policy_act_shape)
            # This part becomes tricky if Q is directly an IQN. The current IQN.next_obs_act_select is more specific.
            # For a generic Q that might be an IQN, we rely on IQN overriding this.
            # If this is called by a simple Q or DoubleQ (non-IQN):
            if q_vals_next_structured.ndim == 3:  # (B, action_dims, num_bins_per_dim)
                max_idcs = torch.argmax(
                    q_vals_next_structured, dim=2
                )  # (batch, action_dims)
            elif (
                q_vals_next_structured.ndim == 4
            ):  # (B, N_tau, action_dims, num_bins_per_dim)
                # This case should ideally be handled by IQN's own next_obs_act_select
                # which averages over N_tau first. If we must handle it here, we'd average or take max over N_tau.
                # For safety, let's assume this path (ndim==4) means it's from an IQN-like policy
                # and it needs to be averaged first if not already done by a wrapper.
                # Given IQN overrides next_obs_act_select, this path is less critical for IQN itself.
                max_idcs = torch.argmax(
                    q_vals_next_structured.mean(dim=1), dim=2
                )  # (batch, action_dims)
            else:
                raise ValueError(
                    "Shape error in next_obs_act_select for independent_dims"
                )

            return max_idcs, q_vals_next_flat
        else:  # joint
            max_idcs = torch.argmax(
                q_vals_next_flat, dim=-1, keepdim=True
            )  # (batch, 1)
            return max_idcs, q_vals_next_flat

    def next_obs_act_eval(
        self, max_idcs, next_obs, q_vals_next_eval=None, use_target_net=True
    ):
        if (
            q_vals_next_eval is None
        ):  # q_vals_next_eval is q_vals_next_flat from select step
            q_vals_next_eval_flat = self.q_pred_next_state(
                next_obs, use_target_net=use_target_net
            )
        else:
            q_vals_next_eval_flat = q_vals_next_eval

        if self.discretization_method == "independent_dims":
            # max_idcs is (batch, action_dims) - chosen bin indices
            # q_vals_next_eval_flat is (batch, policy_act_shape) or (B, N_tau, policy_act_shape)
            original_shape = q_vals_next_eval_flat.shape
            batch_size = original_shape[0]
            num_leading_dims = q_vals_next_eval_flat.ndim - 1

            if num_leading_dims == 1:  # (B, policy_act_shape)
                q_vals_next_eval_structured = q_vals_next_eval_flat.view(
                    batch_size, self.action_dims, self.num_bins_per_dim
                )
                # max_idcs for gather needs to be (B, action_dims, 1)
                max_idcs_expanded = max_idcs.unsqueeze(-1).long()
                # Gather from (B, action_dims, num_bins)
                selected_q_vals_per_dim = q_vals_next_eval_structured.gather(
                    2, max_idcs_expanded
                ).squeeze(-1)  # (batch, action_dims)
            elif num_leading_dims == 2:  # (B, N_tau, policy_act_shape)
                num_tau_dim = original_shape[1]
                q_vals_next_eval_structured = q_vals_next_eval_flat.view(
                    batch_size, num_tau_dim, self.action_dims, self.num_bins_per_dim
                )
                # max_idcs (B, action_dims) needs to be (B, 1, action_dims, 1) to gather from (B, N_tau, action_dims, num_bins)
                max_idcs_expanded = max_idcs.unsqueeze(1).unsqueeze(-1).long()
                # Gather from dim 3 (num_bins_per_dim)
                selected_q_vals_per_dim = q_vals_next_eval_structured.gather(
                    3, max_idcs_expanded
                ).squeeze(-1)  # (batch, N_tau, action_dims)
            else:
                raise ValueError(
                    f"Unexpected shape for q_vals_next_eval_flat: {original_shape}"
                )

            # Sum Q-values across dimensions
            q_vals_next_sum = selected_q_vals_per_dim.sum(
                dim=-1
            )  # sum over action_dims -> (batch,) or (batch, N_tau)
            return q_vals_next_sum
        else:  # joint
            # max_idcs is (batch, 1) - index of the single best joint action
            # q_vals_next_eval_flat is (batch, num_total_discrete_actions)
            q_vals_next = q_vals_next_eval_flat.gather(dim=-1, index=max_idcs).squeeze()
            return q_vals_next

    @torch.no_grad()
    def q_pred_next_state(self, next_obs, use_target_net=True, net=None):
        """Specified in extra method to be potentially overridden by subclasses"""
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
        super().__init__(obs_size, act_size, *args, net=net, **kwargs)

    def __str__(self):
        return f"V <{super().__str__()}>"

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
            del kwargs["dueling"]
        return kwargs
