import torch


def unsqueeze_to(x, target):
    """Adds a dimension to the end of x until x has the same number of dimensions as target"""
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def create_wrapper(baseclass, superclass, add_superclass=None):
    name = f'{str(baseclass)} <{str(superclass)}>'
    if add_superclass is None:
        add_superclass = type
    baseclass_vars = dict(vars(baseclass))
    new_class = add_superclass(name, (baseclass, superclass), baseclass_vars)
    return new_class


def polyak_update(net, target_net, factor):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(factor * target_param.data + param.data * (1.0 - factor))


def copy_weights(source_net, target_net):
    target_net.load_state_dict(source_net.state_dict())


def freeze_params(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors, k=1.0):
    """Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class Standardizer(torch.nn.Module):
    def __init__(self, record_steps):
        """Calculate running mean of first record_steps observations using Welford's method and
        apply z-standardization to observations"""
        super().__init__()
        self.record_steps = record_steps
        self.register_buffer("mean", torch.tensor(0))
        self.register_buffer("std", torch.tensor(1))
        self.register_buffer("run_var", torch.tensor(0))
        self.register_buffer("n", torch.tensor(0))

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
            var = self.run_var / (self.n - 1) if self.n > 1 else torch.tensor(1).type_as(obs)
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
    zipped = zip(tensor_dict.keys(), tensor_dict.values())
    return {
        key: apply_rec_to_dict(func, content) if isinstance(content, dict)
        else func(content)
        for key, content in zipped
    }
