import torch
from torch import nn
from .unit import ResidualUnit
import inspect


class ResidualBlock(nn.Module):
    """
    Residual Block Module
    """

    def __init__(
        self,
        channels,
        activation=nn.LeakyReLU(0.01),
        num_layers=3,
        dilation="3**layer",
        bias=False,
        residual=False,
    ):
        super().__init__()
        self.residual = nn.ModuleList(
            [
                ResidualUnit(
                    channels=channels,
                    activation=activation,
                    dilation=eval(dilation),
                    bias=bias,
                )
                for layer in range(num_layers)
            ]
        )
        self.is_residual = residual

    def forward(self, x):
        out = x
        for layer in self.residual:
            out = layer(out)
        if self.is_residual:
            out = out + x
        return out


class DenseBlock(nn.Module):
    """
    Dense Block Module
    """

    def __init__(
        self,
        channels,
        activation=nn.LeakyReLU(0.01),
        growth_rate=8,
        num_layers=5,
        dilation="2*layer+1",
        bias=False,
        residual=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels + layer * growth_rate,
                        out_channels=growth_rate,
                        kernel_size=3,
                        padding="same",
                        dilation=eval(dilation),
                        bias=bias,
                        padding_mode="reflect",
                    ),
                    activation,
                )
                for layer in range(num_layers)
            ]
        )
        self.transition = nn.Conv1d(
            channels + num_layers * growth_rate,
            channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.residual = residual

    def forward(self, x):
        inputs = x
        for layer in self.layers:
            out = layer(inputs)
            inputs = torch.cat([inputs, out], dim=1)
        out = self.transition(inputs)
        if self.residual:
            out = out + x
        return out


class IAFDenseBlock(nn.Module):
    """
    Dense Block with per-layer IAF:
    after each growth conv, apply affine transform y' = s(prev) * y + t(prev),
    where s, t are produced from all previous channels via 1x1 convs.
    """

    def __init__(
        self,
        channels,
        activation=nn.LeakyReLU(0.01),
        growth_rate=None,
        num_layers=5,
        dilation="2*layer+1",
        bias=False,
        residual=False,
        scale_clamp=2.0,
        eps=1e-6,
    ):
        super().__init__()
        if growth_rate is None:
            growth_rate = channels // num_layers
        self.eps = eps
        self.scale_clamp = scale_clamp
        self.residual = residual
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels + layer * growth_rate,
                        out_channels=growth_rate,
                        kernel_size=3,
                        padding="same",
                        dilation=eval(dilation),
                        bias=bias,
                        padding_mode="reflect",
                    ),
                    activation,
                )
                for layer in range(num_layers)
            ]
        )
        self.cond_scale = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels + layer * growth_rate,
                    out_channels=growth_rate,
                    kernel_size=1,
                    padding="same",
                    bias=bias,
                    padding_mode="reflect",
                )
                for layer in range(num_layers)
            ]
        )
        self.cond_shift = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=channels + layer * growth_rate,
                    out_channels=growth_rate,
                    kernel_size=1,
                    padding="same",
                    bias=bias,
                    padding_mode="reflect",
                )
                for layer in range(num_layers)
            ]
        )
        self.transition = nn.Conv1d(
            in_channels=channels + num_layers * growth_rate,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, x):
        inputs = x
        for l, layer in enumerate(self.layers):
            y = layer(inputs)
            s_raw = self.cond_scale[l](inputs)
            t = self.cond_shift[l](inputs)
            s = torch.exp(torch.tanh(s_raw) * self.scale_clamp) + self.eps
            y = s * y + t
            inputs = torch.cat([inputs, y], dim=1)

        out = self.transition(inputs)
        if self.residual:
            out = out + x
        return out


BASEBLOCK = {name.lower(): obj for name, obj in globals().items() if inspect.isclass(obj) and issubclass(obj, nn.Module)}
