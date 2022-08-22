# From https://vsitzmann.github.io/siren/ (MIT License)
import numpy as np
import torch
from torch import nn


class SineConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Conv2d(in_features, out_features, 1, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1. / self.in_features,
                                            1. / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6. / self.in_features) / self.omega_0,
                                            np.sqrt(6. / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SirenConv(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        assert hidden_layers > 0
        self.net = []
        self.net.append(SineConvLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers - 1):
            self.net.append(SineConvLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Conv2d(hidden_features, out_features, 1)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6. / hidden_features) / hidden_omega_0,
                                             np.sqrt(6. / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineConvLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)
