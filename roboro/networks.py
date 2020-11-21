import torch
import numpy as np

from roboro.layers import create_block, DuelingLayer


class CNN(torch.nn.Module):
    def __init__(self, input_shape):
        """
        Args:
            input_shape: observation shape of the environment
        """
        super().__init__()
        in_channels = input_shape[0]
        module_list = [torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                       torch.nn.ReLU(True),
                       torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                       torch.nn.ReLU(True),
                       torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                       torch.nn.ReLU(True)]
        self.conv = torch.nn.Sequential(*module_list)

        self.conv_out_size = self.get_conv_out_size(input_shape)

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
        return conv_out

    def get_out_size(self):
        return self.conv_out_size


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, dueling=False, noisy=False, width=512):
        super().__init__()
        self.out_size = out_size
        linear_kwargs = {"noisy_linear": noisy, "width": width}
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
