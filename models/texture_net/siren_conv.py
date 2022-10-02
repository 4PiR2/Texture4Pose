# From https://vsitzmann.github.io/siren/ (MIT License)
import numpy as np
import torch
from torch import nn

# from utils.image_2d import visualize, show_tensor_hist


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
                # for input uniform(-r, r), init should be uniform(-3 / (r * sqrt(n)), 3 / (r * sqrt(n)))
            else:
                self.linear.weight.uniform_(-np.sqrt(6. / self.in_features) / self.omega_0,
                                            np.sqrt(6. / self.in_features) / self.omega_0)

    def forward(self, input, omega=None):
        if omega is None:
            omega = self.omega_0
        return torch.sin(omega * self.linear(input))


class SirenConv(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=1., hidden_omega_0=1.):
        super().__init__()
        assert hidden_layers > 0
        layers = []
        layers.append(SineConvLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers - 1):
            layers.append(SineConvLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Conv2d(hidden_features, out_features, 1)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6. / hidden_features) / hidden_omega_0,
                                             np.sqrt(6. / hidden_features) / hidden_omega_0)
            layers.append(final_linear)
        else:
            layers.append(SineConvLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.first_layer = layers[0]
        self.rest_layers = nn.Sequential(*layers[1:])

    def forward(self, x):
        x = self.first_layer(x)
        x = self.rest_layers(x)
        # show_tensor_hist(self.net[0].linear.weight, bins=50)
        return (x * .5 + .5).clamp(min=0., max=1.)
