# quaternion_layers.py
"""Quaternion Conv/BN helpers for PyTorch (1-pass conv version).

Tensor layout:
    (B, 4*C, L)   where the four consecutive channels are
    [r0, i0, j0, k0,  r1, i1, j1, k1,  ...]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def real4_to_quat(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split (B,4C,L) into 4× (B,C,L)."""
    Br4L = x.shape
    assert len(Br4L) == 3 and Br4L[1] % 4 == 0, "Channel must be multiple of 4"
    return x.chunk(4, dim=1)  # r, i, j, k


def quat_to_real4(r: Tensor, i: Tensor, j: Tensor, k: Tensor) -> Tensor:
    """Concat 4× (B,C,L) back to (B,4C,L)."""
    return torch.cat([r, i, j, k], dim=1)


class QuaternionConv1d(nn.Module):
    """
    Quaternion 1-D convolution implemented with *one* real Conv1d.
    The 4×4 Hamilton product pattern is encoded into a big weight
    tensor constructed on-the-fly every forward pass.

    Args
    ----
    in_channels  :   # quaternion input channels (C_in)
    out_channels :   # quaternion output channels (C_out)
    kernel_size  :   int
    stride       :   int or tuple
    padding      :   int | tuple | 'same'
    dilation     :   int or tuple
    bias         :   bool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        ks = kernel_size

        # Four shared real weight groups:  (C_out, C_in, K)
        shp = (out_channels, in_channels, ks)
        self.Wr = nn.Parameter(torch.empty(shp))
        self.Wi = nn.Parameter(torch.empty(shp))
        self.Wj = nn.Parameter(torch.empty(shp))
        self.Wk = nn.Parameter(torch.empty(shp))

        if bias:
            # One bias per *real* output channel (4*C_out)
            self.bias = nn.Parameter(torch.zeros(4 * out_channels))
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.reset_parameters()

    # ----------------------------------------
    def reset_parameters(self):
        """Quaternion He init (same for all 4 groups)."""
        for W in (self.Wr, self.Wi, self.Wj, self.Wk):
            nn.init.kaiming_uniform_(W, a=0, mode="fan_in", nonlinearity="relu")

    # ----------------------------------------
    @staticmethod
    def _cat_weights(Wr, Wi, Wj, Wk) -> Tensor:
        """Build (4*C_out, 4*C_in, K) weight tensor following Hamilton rules."""
        # Rows:   [out_r, out_i, out_j, out_k]
        # Colums: [ in_r,  in_i,  in_j,  in_k]
        #
        # Each block is  (C_out, C_in, K)
        # Signs follow:
        #   out_r =  +Wr @ in_r  -Wi @ in_i  -Wj @ in_j  -Wk @ in_k
        #   out_i =  +Wi @ in_r  +Wr @ in_i  +Wk @ in_j  -Wj @ in_k
        #   out_j =  +Wj @ in_r  -Wk @ in_i  +Wr @ in_j  +Wi @ in_k
        #   out_k =  +Wk @ in_r  +Wj @ in_i  -Wi @ in_j  +Wr @ in_k

        A = torch.cat  # alias

        row_r = A([+Wr, -Wi, -Wj, -Wk], dim=1)
        row_i = A([+Wi, +Wr, +Wk, -Wj], dim=1)
        row_j = A([+Wj, -Wk, +Wr, +Wi], dim=1)
        row_k = A([+Wk, +Wj, -Wi, +Wr], dim=1)
        W_big = A([row_r, row_i, row_j, row_k], dim=0)  # (4*C_out, 4*C_in, K)
        return W_big

    # ----------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, 4*C_in, L)
        y : (B, 4*C_out, L)
        """
        # Build big weight just-in-time (no grad duplication, cheap on GPU)
        W_big = self._cat_weights(self.Wr, self.Wi, self.Wj, self.Wk)

        y = F.conv1d(
            x,
            W_big,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)

        return y

class QuaternionBatchNorm1d(nn.Module):
    """
    Per-component (r/i/j/k) BatchNorm. Optionally share γ,β across components
    to keep magnitude consistency.
    """

    def __init__(
        self,
        num_features: int,  # quaternion channels
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        shared_affine: bool = False,
    ):
        super().__init__()
        self.num_feat = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.shared = shared_affine

        self.register_buffer("running_mean", torch.zeros(4 * num_features))
        self.register_buffer("running_var", torch.ones(4 * num_features))

        if affine:
            if shared_affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.weight = nn.Parameter(torch.ones(4 * num_features))
                self.bias = nn.Parameter(torch.zeros(4 * num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    # ----------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        B, C4, L = x.shape
        assert C4 == 4 * self.num_feat, "Mismatched channel size"

        if self.training:
            mean = x.mean(dim=(0, 2))
            var = x.var(dim=(0, 2), unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean.view(1, -1, 1)) / (var.view(1, -1, 1) + self.eps).sqrt()

        if self.affine:
            if self.shared:
                w = self.weight.repeat_interleave(4)
                b = self.bias.repeat_interleave(4)
            else:
                w, b = self.weight, self.bias
            x_hat = x_hat * w.view(1, -1, 1) + b.view(1, -1, 1)

        return x_hat

