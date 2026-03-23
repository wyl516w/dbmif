import torch
import torch.nn as nn
from typing import Tuple


def make_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    bias: bool = False,
) -> nn.Conv1d:
    """Create a 1‑D convolution with ‘same’ padding."""
    padding = kernel_size // 2
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
    )


class GateConvSideFusion(nn.Module):
    """
    One‑direction gated convolutional fusion block.

        out = activation( base_conv(x) * sigmoid(gated_conv(y)) + residual(x) )

    Parameters
    ----------
    in1_channels : int
        Number of channels in the source tensor `x`.
    in2_channels : int
        Number of channels in the context tensor `y`.
    out_channels  : int
        Number of output channels produced by this branch.
    kernel_size   : int
        Kernel size for interaction convolutions.
    bias          : bool, default=False
        Whether to use bias terms in convolution layers.
    residual      : bool, default=False
        If True, adds a residual pathway (1×1 conv or identity).
    activation    : nn.Module, default=nn.Tanh()
        Non‑linear function applied at the end.
    """

    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
        residual: bool = False,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()

        # Base convolution on x.
        self.base_conv = make_conv1d(
            in_channels=in1_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

        # Gating convolution on y.
        self.gated_conv = make_conv1d(
            in_channels=in2_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

        # Optional residual mapping.
        if residual:
            self.residual_map = (
                nn.Identity()
                if in1_channels == out_channels
                else nn.Conv1d(
                    in_channels=in1_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )
        else:
            self.residual_map = None

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, C_x, T)
            Source feature map.
        y : Tensor, shape (B, C_y, T)
            Context feature map supplying gating information.

        Returns
        -------
        Tensor, shape (B, C_out, T)
            Fused and activated output.
        """
        base_out = self.base_conv(x)
        gate = self.sigmoid(self.gated_conv(y))
        fused = base_out * gate  # element‑wise modulation

        if self.residual_map is not None:
            fused = fused + self.residual_map(x)

        return self.activation(fused)


class GateConvFusion(nn.Module):
    """
    Symmetric two‑way gated fusion block.

    Computes:
        out_x = F(x ← y)
        out_y = F(y ← x)
    """

    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out1_channels: int,
        out2_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
        residual: bool = False,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()

        self.side1 = GateConvSideFusion(
            in1_channels,
            in2_channels,
            out1_channels,
            kernel_size,
            bias=bias,
            residual=residual,
            activation=activation,
        )

        self.side2 = GateConvSideFusion(
            in2_channels,
            in1_channels,
            out2_channels,
            kernel_size,
            bias=bias,
            residual=residual,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x, y : Tensor
            Input feature maps.

        Returns
        -------
        out_x, out_y : Tensor
            Fusion outputs for x and y respectively.
        """
        out_x = self.side1(x, y)  # x fused with context y
        out_y = self.side2(y, x)  # y fused with context x
        return out_x, out_y


class SplitGateConvFusion(nn.Module):
    """
    Two‑way gated fusion with a single Conv per stream.

    For each input stream (x / y), one Conv1d outputs 2×out_channels:
      • first  half → base features for itself
      • second half → gate features for the opposite stream

    Calculations
    ------------
        out_x = act( base_x * σ(gate_xy) + residual_x )
        out_y = act( base_y * σ(gate_yx) + residual_y )
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        *,
        bias: bool = False,
        residual: bool = False,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()

        # Each stream has ONE convolution whose output channel dim = 2×out_channels
        padding = kernel_size // 2
        self.conv_x = nn.Conv1d(
            in_channels=channels_in,
            out_channels=channels_out * 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_y = nn.Conv1d(
            in_channels=channels_in,
            out_channels=channels_out * 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # Optional residual pathways (identity if shapes match)
        if residual:
            self.res_x = nn.Identity() if channels_in == channels_out else nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False)
            self.res_y = nn.Identity() if channels_in == channels_out else nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False)
        else:
            self.res_x = self.res_y = None

        self.sigmoid = nn.Sigmoid()
        self.activation = activation

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x, y : Tensor (B, C_in, T)
            Input feature maps.

        Returns
        -------
        out_x, out_y : Tensor (B, C_out, T)
            Fusion outputs for x and y respectively.
        """
        # -------- Conv & split for x -------- #
        feat_x = self.conv_x(x)  # (B, 2*C_out, T)
        base_x, gate_yx = feat_x.chunk(2, dim=1)  # first half, second half

        # -------- Conv & split for y -------- #
        feat_y = self.conv_y(y)  # (B, 2*C_out, T)
        base_y, gate_xy = feat_y.chunk(2, dim=1)

        # -------- x ← y fusion -------- #
        out_x = base_x * self.sigmoid(gate_xy)
        if self.res_x is not None:
            out_x = out_x + self.res_x(x)
        out_x = self.activation(out_x)

        # -------- y ← x fusion -------- #
        out_y = base_y * self.sigmoid(gate_yx)
        if self.res_y is not None:
            out_y = out_y + self.res_y(y)
        out_y = self.activation(out_y)

        return out_x, out_y
