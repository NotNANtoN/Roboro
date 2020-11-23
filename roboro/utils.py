import torch


class Standardizer(torch.nn.Module):
    def __init__(self, record_steps):
        """Calculate running mean of first record_steps observations using Welford's method and
        apply z-standardization to observations"""
        super().__init__()
        self.record_steps = record_steps
        self.register_buffer("mean", torch.tensor(0))
        self.register_buffer("std", torch.tensor(0))
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
            var = self.run_var / (self.n - 1) if self.n > 1 else torch.tensor(1)
            self.std = torch.sqrt(var) if self.n > 1 else torch.tensor(1)
            self.mean = new_mean

    def norm(self, obs):
        standardized_obs = (obs - self.mean) / self.std
        # if np.isnan(standardized_obs).sum():
        #    print(obs)
        #    print(standardized_obs)
        #    print(self.mean)
        #    print(self.std)
        #    quit()
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
