from torch import nn
import torch


class ConvSideFusion(nn.Module):
    def __init__(
        self,
        in1_channels,
        in2_channels,
        out_channels,
        kernel_size,
        bias=False,
        residual=False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.conv = nn.Conv1d(in1_channels + in2_channels, out_channels, kernel_size, bias=bias, padding="same")
        self.activation = activation

    def forward(self, x, y):
        out = self.conv(torch.cat((x, y), dim=1))
        return self.activation(out)


class ConvFusion(nn.Module):
    def __init__(
        self,
        in1_channels,
        in2_channels,
        out1_channels,
        out2_channels,
        kernel_size,
        bias=False,
        residual=False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.side1 = ConvSideFusion(
            in1_channels,
            in2_channels,
            out1_channels,
            kernel_size,
            bias,
            residual,
            activation,
        )
        self.side2 = ConvSideFusion(
            in2_channels,
            in1_channels,
            out2_channels,
            kernel_size,
            bias,
            residual,
            activation,
        )

    def forward(self, x, y):
        return self.side1(x, y), self.side2(y, x)
