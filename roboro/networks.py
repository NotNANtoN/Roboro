import torch
import numpy as np

from roboro.layers import create_dense_layer, DuelingLayer


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
        self.in_to_hidden = create_dense_layer(in_size, width, **linear_kwargs)
        if dueling:
            self.hidden_to_out = DuelingLayer(width, out_size, **linear_kwargs)
        else:
            self.hidden_to_out = create_dense_layer(width, out_size, act_func=False, **linear_kwargs)

    def forward(self, state_features):
        hidden = self.in_to_hidden(state_features)
        out = self.hidden_to_out(hidden)
        return out

    def get_out_size(self):
        return self.out_size


class IQNNet(torch.nn.Module):
    """IQN net. Adapted from https://github.com/BY571/IQN-and-Extensions"""
    def __init__(self, obs_size, act_size, num_tau, num_policy_samples, width=512, dueling=False, noisy_layers=False):
        super().__init__()
        self.in_size = obs_size
        self.state_dim = 1
        self.action_size = act_size
        self.num_tau = num_tau
        self.num_policy_samples = num_policy_samples
        self.n_cos = 64
        self.layer_size = width

        # Starting from 0 as in the paper
        self.register_buffer("pis", torch.tensor([np.pi * i for i in range(1, self.n_cos + 1)],
                                                 dtype=torch.float).view(1, 1, self.n_cos))
        # Network Architecture
        # embedding layers
        self.head = torch.nn.Linear(self.in_size, width)
        self.cos_layer_out = width
        self.cos_embedding = torch.nn.Linear(self.n_cos, self.cos_layer_out)
        # processing layers
        self.out_mlp = MLP(width, act_size, dueling=dueling, noisy_layers=noisy_layers, width=width)

    def forward(self, obs, num_quantiles=None):
        if num_quantiles is None:
            num_quantiles = self.num_policy_samples
        quantiles, _ = self.get_quantiles(obs, num_quantiles)
        actions = quantiles.mean(dim=1)
        return actions

    def calc_input_layer(self):
        x = torch.zeros(self.in_size).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]

    def sample_cos(self, batch_size, num_tau=None):
        """
        Calculating the cosine values depending on the number of tau samples
        """
        if num_tau is None:
            num_tau = self.num_tau
        taus = torch.rand(batch_size, num_tau, 1, dtype=self.pis.dtype, device=self.pis.device)
        # shape is (batch_size, num_tau, 1)
        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, num_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def get_quantiles(self, obs, num_tau=None, cos=None, taus=None):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        if num_tau is None:
            num_tau = self.num_tau
        batch_size = obs.shape[0]
        if cos is None:
            cos, taus = self.sample_cos(batch_size, num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)

        x = torch.relu(self.head(obs))
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau,
                                                         self.cos_layer_out)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.cos_layer_out)

        x = self.out_mlp(x)

        return x.view(batch_size, num_tau, self.action_size), taus
