from torch import nn
import torch 

class ConvAttention(nn.Module):
    def __init__(
        self,
        in1_channels,
        in2_channels,
        out_channels,
        kernel_size,
        inter_channels,
        bias=False,
    ):
        super().__init__()
        self.wq = nn.Conv1d(
            in1_channels,
            inter_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.wk = nn.Conv1d(
            in2_channels,
            inter_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.wv = nn.Conv1d(
            in2_channels,
            inter_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.final_conv = nn.Conv1d(
            inter_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.inter_channels = inter_channels

    def forward(self, x, y):
        # x,y (batch_size, in_channels, time_len)
        Q = self.wq(x)  # (batch_size, inter_channels, time_len)
        K = self.wk(y)  # (batch_size, inter_channels, time_len)
        V = self.wv(y)  # (batch_size, inter_channels, time_len)
        # Q * K (batch_size, inter_channels, inter_channels)
        attention = torch.bmm(Q, K.transpose(1, 2))
        # normalize
        attention = attention / (self.inter_channels**0.5)
        # softmax
        attention = torch.nn.functional.softmax(attention, dim=2)
        # attention * V (batch_size, inter_channels, time_len)
        attention = torch.bmm(attention, V)
        out = self.final_conv(attention)
        return out

class ConvattSideFusion(nn.Module):
    def __init__(
        self,
        in1_channels,
        in2_channels,
        out_channels,
        kernel_size,
        inter_channels,
        bias=False,
        residual=False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.attention = ConvAttention(
            in1_channels,
            in2_channels,
            out_channels,
            kernel_size,
            inter_channels,
            bias=bias,
        )
        self.residual = None
        if residual:
            self.residual = nn.Conv1d(in_channels=in1_channels, out_channels=out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x, y):
        out = self.attention(x, y)
        if self.residual is not None:
            out = out + self.residual(x)
        return self.activation(out)

class ConvattFusion(nn.Module):
    def __init__(
        self,
        in1_channels,
        in2_channels,
        out1_channels,
        out2_channels,
        kernel_size,
        inter_channels,
        bias=False,
        residual=False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.side1 = ConvattSideFusion(
            in1_channels,
            in2_channels,
            out1_channels,
            kernel_size,
            inter_channels,
            bias,
            residual,
            activation,
        )
        self.side2 = ConvattSideFusion(
            in2_channels,
            in1_channels,
            out2_channels,
            kernel_size,
            inter_channels,
            bias,
            residual,
            activation,
        )

    def forward(self, x, y):
        return self.side1(x, y), self.side2(y, x)

