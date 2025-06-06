import torch


def unsqueeze_to(x, target):
    """Adds a dimension to the end of x until x has the same number of dimensions as target"""
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def create_wrapper(baseclass, superclass, add_superclass=None):
    name = f"{str(baseclass)} <{str(superclass)}>"
    if add_superclass is None:
        add_superclass = type
    baseclass_vars = dict(vars(baseclass))
    new_class = add_superclass(name, (baseclass, superclass), baseclass_vars)
    return new_class


def polyak_update(net, target_net, factor):
    for target_param, param in zip(
        target_net.parameters(), net.parameters(), strict=False
    ):
        target_param.data.copy_(
            factor * target_param.data + param.data * (1.0 - factor)
        )


def copy_weights(source_net, target_net):
    target_net.load_state_dict(source_net.state_dict())


def freeze_params(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors, k=1.0):
    """Calculate huber loss element-wisely depending on kappa k."""
    loss = torch.where(
        td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
    )
    return loss


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


class Standardizer(torch.nn.Module):
    def __init__(self, record_steps):
        """Calculate running mean of first record_steps observations using Welford's method and
        apply z-standardization to observations"""
        super().__init__()
        self.record_steps = record_steps
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("run_var", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("n", torch.tensor(0, dtype=torch.int64))

    def observe(self, obs):
        # Check if we want to update the mean and std
        if self.n > self.record_steps:
            return
        # Update mean and std
        self.n += 1
        if self.mean is None:
            self.mean = obs.copy()
            self.run_var = torch.tensor(0)
        else:
            new_mean = self.mean + (obs - self.mean) / self.n
            self.run_var = self.run_var + (obs - new_mean) * (obs - self.mean)
            var = (
                self.run_var / (self.n - 1)
                if self.n > 1
                else torch.tensor(1).type_as(obs)
            )
            self.std = torch.sqrt(var)
            self.mean = new_mean
            # Some inputs features (e.g. pixels) might never change
            self.std[self.std == 0] = 1

    def norm(self, obs):
        standardized_obs = (obs - self.mean) / self.std
        return standardized_obs

    def denorm(self, obs):
        return (obs * self.std) + self.mean


def apply_to_state_list(func, state_list):
    """Apply function to whole list of states (e.g. concatenate, stack etc.)
    Recursively apply this function to nested dicts."""
    if isinstance(state_list[0], dict):
        return {
            key: apply_to_state_list(func, [state[key] for state in state_list])
            for key in state_list[0]
        }
    else:
        return func(state_list)


def apply_to_state(func, state):
    """Apply function recursively to state dict or directly to state"""
    if isinstance(state, dict):
        return apply_rec_to_dict(func, state)
    else:
        return func(state)


def apply_rec_to_dict(func, tensor_dict):
    """Apply a function recursively to every non dict object in a nested dict"""
    zipped = zip(tensor_dict.keys(), tensor_dict.values(), strict=False)
    return {
        key: (
            apply_rec_to_dict(func, content)
            if isinstance(content, dict)
            else func(content)
        )
        for key, content in zipped
    }


def map_bin_indices_to_continuous_tensor(
    bin_indices_batch: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_bins_per_dim: int,
) -> torch.Tensor:
    """
    Converts a batch of bin index vectors to continuous action vectors.

    Args:
        bin_indices_batch: Tensor of shape (batch_size, action_dims) with chosen bin indices.
        action_low: Tensor of shape (action_dims,) with lower bounds for each action dimension.
        action_high: Tensor of shape (action_dims,) with upper bounds for each action dimension.
        num_bins_per_dim: Number of bins per action dimension.

    Returns:
        Tensor of shape (batch_size, action_dims) with continuous action values.
    """
    action_low = action_low.to(bin_indices_batch.device)
    action_high = action_high.to(bin_indices_batch.device)

    # Ensure bin_indices are float for division
    bin_indices_float = bin_indices_batch.float()

    if num_bins_per_dim == 1:
        # If only one bin, action is the midpoint
        continuous_actions = (action_low + action_high) / 2.0
        # Expand to batch size if necessary
        continuous_actions = continuous_actions.unsqueeze(0).expand(
            bin_indices_batch.shape[0], -1
        )
    else:
        # normalized_values = bin_indices / (num_bins_per_dim - 1)
        # continuous_actions = action_low + normalized_values * (action_high - action_low)
        # Reshape for broadcasting:
        # action_low, action_high: (action_dims,) -> (1, action_dims)
        # bin_indices_float: (batch_size, action_dims)
        normalized_values = bin_indices_float / (num_bins_per_dim - 1)
        continuous_actions = action_low.unsqueeze(0) + normalized_values * (
            action_high.unsqueeze(0) - action_low.unsqueeze(0)
        )
    return continuous_actions


def map_continuous_to_bin_indices_tensor(
    continuous_actions_batch: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    num_bins_per_dim: int,
) -> torch.Tensor:
    """
    Converts a batch of continuous action vectors to bin index vectors.

    Args:
        continuous_actions_batch: Tensor of shape (batch_size, action_dims).
        action_low: Tensor of shape (action_dims,) with lower bounds.
        action_high: Tensor of shape (action_dims,) with upper bounds.
        num_bins_per_dim: Number of bins per action dimension.

    Returns:
        Tensor of shape (batch_size, action_dims) with chosen bin indices (long).
    """
    action_low = action_low.to(continuous_actions_batch.device)
    action_high = action_high.to(continuous_actions_batch.device)

    if num_bins_per_dim == 1:
        bin_indices = torch.zeros_like(continuous_actions_batch, dtype=torch.long)
    else:
        # Normalize continuous actions to [0, 1] range
        # Add a small epsilon to handle floating point issues at the boundaries for (high - low)
        range_val = action_high.unsqueeze(0) - action_low.unsqueeze(0)
        # Prevent division by zero if low == high for some dimension
        range_val[range_val == 0] = 1.0

        normalized_actions = (
            continuous_actions_batch - action_low.unsqueeze(0)
        ) / range_val
        # Scale to [0, num_bins_per_dim - 1]
        scaled_actions = normalized_actions * (num_bins_per_dim - 1)
        # Round to nearest bin index and clamp
        bin_indices = torch.round(scaled_actions).long().clamp_(0, num_bins_per_dim - 1)
    return bin_indices
