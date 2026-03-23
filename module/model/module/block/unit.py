import torch
from torch import nn


class LSTM(nn.Module):
    """
    LSTM Module Rewritten for Block Model
    """

    def __init__(self, channels, num_layers=1, bidirectional=False, bias=True, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels // (2 if bidirectional else 1),
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))
        return out.transpose(1, 2)


class ResidualUnit(nn.Module):
    """
    Residual Unit Module
    """

    def __init__(self, channels, activation, dilation, bias=False):
        super().__init__()
        self.dilated_conv = NormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.pointwise_conv = NormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.activation = activation

    def forward(self, x):
        out = x + self.activation(self.pointwise_conv(self.dilated_conv(x)))
        return out


class NormConv1d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class NormConvTrans1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)
