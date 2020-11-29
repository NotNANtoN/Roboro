import torch
import numpy as np

from roboro.layers import create_block, DuelingLayer, NoisyLinear


class CNN(torch.nn.Module):
    def __init__(self, input_shape, feat_size):
        """
        Args:
            input_shape: observation shape of the environment
            feat_size: size of the feature head layer at the end
        """
        super().__init__()
        self.feat_size = feat_size
        in_channels = input_shape[0]
        module_list = [torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                       torch.nn.ReLU(True),
                       torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                       torch.nn.ReLU(True),
                       torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                       torch.nn.ReLU(True)
                       ]
        self.conv = torch.nn.Sequential(*module_list)
        conv_out_size = self.get_conv_out_size(input_shape)
        self.head = torch.nn.Sequential(torch.nn.Linear(conv_out_size, feat_size),
                                        torch.nn.ReLU(True))

    def get_conv_out_size(self, shape) -> int:
        """
        Calculates the output size of the last conv layer
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.shape))

    def forward(self, obs):
        conv_out = self.conv(obs).view(obs.shape[0], -1)
        out = self.head(conv_out)
        return out

    def get_out_size(self):
        return self.feat_size


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, dueling=False, noisy_layers=False, width=512):
        super().__init__()
        self.out_size = out_size
        linear_kwargs = {"noisy_linear": noisy_layers, "width": width}
        self.in_to_hidden = create_block(in_size, width, **linear_kwargs)
        if dueling:
            self.hidden_to_out = DuelingLayer(width, out_size, **linear_kwargs)
        else:
            self.hidden_to_out = create_block(width, out_size, **linear_kwargs)

    def forward(self, state_features):
        hidden = self.in_to_hidden(state_features)
        out = self.hidden_to_out(hidden)
        return out

    def get_out_size(self):
        return self.out_size


class IQNNet(torch.nn.Module):
    """IQN net. Adapted from https://github.com/BY571/IQN-and-Extensions"""
    def __init__(self, obs_size, act_size, num_quantiles,  width=512, dueling=False, noisy_layers=False):
        super().__init__()
        self.in_size = obs_size
        self.state_dim = 1
        self.action_size = act_size
        self.num_quantiles = num_quantiles
        self.n_cos = 64
        self.layer_size = width

        # Starting from 0 as in the paper
        self.register_buffer("pis", torch.tensor([np.pi * i for i in range(1, self.n_cos + 1)],
                                                 dtype=torch.float).view(1, 1, self.n_cos))
        self.dueling = dueling
        if noisy_layers:
            layer = NoisyLinear
        else:
            layer = torch.nn.Linear

        # Network Architecture
        self.head = torch.nn.Linear(self.in_size, width)
        self.cos_embedding = torch.nn.Linear(self.n_cos, width)
        self.ff_1 = layer(width, width)
        self.cos_layer_out = width
        if dueling:
            self.advantage = layer(width, act_size)
            self.value = layer(width, 1)
            # weight_init([self.head_1, self.ff_1])
        else:
            self.ff_2 = layer(width, act_size)
            # weight_init([self.head_1, self.ff_1])

    def forward(self, obs):
        quantiles, _ = self.get_quantiles(obs, self.num_quantiles)
        actions = quantiles.mean(dim=1)
        return actions

    def calc_input_layer(self):
        x = torch.zeros(self.in_size).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau, 1, dtype=self.pis.dtype, device=self.pis.device)  # (batch_size, n_tau, 1)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def get_quantiles(self, obs, num_tau=8):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        batch_size = obs.shape[0]

        x = torch.relu(self.head(obs))
        if self.state_dim == 3:
            x = x.view(obs.size(0), -1)
        cos, taus = self.calc_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau,
                                                         self.cos_layer_out)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.cos_layer_out)

        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)

        return out.view(batch_size, num_tau, self.action_size), taus
