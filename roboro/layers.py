import math

import torch


def create_block(in_shape, out_shape, width=512, noisy_linear=False):
    if noisy_linear:
        create_linear = lambda fan_in, fan_out: NoisyLinear(fan_in, fan_out)
    else:
        create_linear = lambda fan_in, fan_out: torch.nn.Linear(fan_in, fan_out)
    module_list = [create_linear(in_shape, width),
                   torch.nn.ReLU(True),
                   create_linear(width, out_shape)]
    modules = torch.nn.Sequential(*module_list)
    return modules


class DuelingLayer(torch.nn.Module):
    """
    MLP network with duel heads for val and advantage
    """

    def __init__(self, in_size, out_size, **linear_kwargs):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.head_adv = create_block(in_size, out_size, **linear_kwargs)
        self.head_val = create_block(in_size, 1, **linear_kwargs)

    def forward(self, x):
        """
        Forward pass through layer. Calculates the Q using the value and advantage
        Args:
            x: input to network
        Returns:
            Q value
        """
        adv = self.head_adv(x)
        val = self.head_val(x)
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val


class NoisyLinear(torch.nn.Linear):
    """
    Noisy Layer using Independent Gaussian Noise.
    based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/
    Chapter08/lib/dqn_extra.py#L19
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        """
        Args:
            in_features: number of inputs
            out_features: number of outputs
            sigma_init: initial fill value of noisy weights
            bias: flag to include bias to linear layer
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        weights = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = torch.nn.Parameter(weights)
        epsilon_weight = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight)

        if bias:
            bias = torch.full((out_features,), sigma_init)
            self.sigma_bias = torch.nn.Parameter(bias)
            epsilon_bias = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initializes or resets the paramseter of the layer"""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x):
        """
        Forward pass of the layer
        Args:
            input_x: input tensor
        Returns:
            output of the layer
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        noisy_weights = self.sigma_weight * self.epsilon_weight.data + self.weight
        return torch.nn.functional.linear(input_x, noisy_weights, bias)
