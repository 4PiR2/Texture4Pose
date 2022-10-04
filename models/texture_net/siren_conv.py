# From https://vsitzmann.github.io/siren/ (MIT License)
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
        self.linear = nn.Conv2d(in_features, out_features, 1, bias=bias and not is_first)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # bound = 1. / self.in_features
                bound = 3. / torch.tensor(self.in_features).sqrt()
                # for input uniform(-r, r), init should be uniform(-3 / (r * sqrt(n)), 3 / (r * sqrt(n)))
            else:
                bound = torch.tensor(6. / self.in_features).sqrt() / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, input, omega=None):
        if omega is None:
            omega = self.omega_0
        return torch.sin(.5 * torch.pi * omega * self.linear(input))


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
                bound = torch.tensor(6. / hidden_features).sqrt() / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
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
