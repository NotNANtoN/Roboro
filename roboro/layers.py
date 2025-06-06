import math

import torch


def get_activation(activation_name: str = "relu") -> torch.nn.Module:
    """Returns the activation function based on the name"""
    if activation_name.lower() == "mish":
        return torch.nn.Mish()
    elif activation_name.lower() == "relu":
        return torch.nn.ReLU()
    return torch.nn.ReLU(True)  # Default to ReLU


def create_dense_layer(
    in_size: int,
    out_size: int,
    noisy_linear: bool = False,
    act_func: str | None = None,
    use_layer_norm: bool = False,
    dropout_rate: float | None = None,
) -> torch.nn.Module:
    """Creates a dense layer with optional layer normalization and dropout

    Args:
        in_size: input size
        out_size: output size
        noisy_linear: whether to use noisy linear layer
        act_func: activation function name or None
        use_layer_norm: whether to use layer normalization
        dropout_rate: dropout rate (None for no dropout)
    """
    create_linear = NoisyLinear if noisy_linear else torch.nn.Linear
    module_list = [create_linear(in_size, out_size)]

    if use_layer_norm:
        module_list.append(torch.nn.LayerNorm(out_size))

    if dropout_rate is not None and dropout_rate > 0:
        module_list.append(torch.nn.Dropout(p=dropout_rate))

    if act_func:
        module_list.append(get_activation(act_func))

    modules = torch.nn.Sequential(*module_list)
    return modules


class DuelingLayer(torch.nn.Module):
    """
    MLP network with duel heads for val and advantage
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        v_head: torch.nn.Module | None = None,
        **linear_kwargs,
    ):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.head_adv = create_dense_layer(
            in_size, out_size, act_func=False, **linear_kwargs
        )
        if v_head is None:
            self.head_val = create_dense_layer(
                in_size, 1, act_func=False, **linear_kwargs
            )
        else:
            self.head_val = v_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.017,
        bias: bool = True,
    ):
        """
        Args:
            in_features: number of inputs
            out_features: number of outputs
            sigma_init: initial fill value of noisy weights
            bias: flag to include bias to linear layer
        """
        super().__init__(in_features, out_features, bias=bias)

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

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
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
