import torch
from torch import nn
from torch.nn import functional as F
from .unit import NormConv1d, NormConvTrans1d
from .base import BASEBLOCK
import inspect


class EncBlock(nn.Module):
    """
    Encoder Block Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        activation,
        block={
            "method": "residualblock",
            "num_layers": 3,
            "dilation": "3**layer",
            "residual": False,
            "bias": False,
        },
    ):
        super().__init__()
        method = block.get("method", "residualblock")
        bias = block.get("bias", False)
        args = dict(filter(lambda x: x[0] not in ["method"], block.items()))
        self.activation = activation
        if method in BASEBLOCK:
            self.block = BASEBLOCK[method](
                channels=in_channels,
                activation=activation,
                **args,
            )
        else:
            assert False, f"Unknown block type: {method}"

        self.conv = NormConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride - 1,
            padding_mode="reflect",
            bias=bias,
        )

    def forward(self, x):
        out = self.conv(self.block(self.activation((x))))
        return out


class DecBlock(nn.Module):
    """
    Decoder Block Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        activation,
        block={
            "method": "residualblock",
            "num_layers": 3,
            "dilation": "3**layer",
            "residual": False,
            "bias": False,
        },
    ):
        super().__init__()
        method = block.get("method", "residualblock")
        bias = block.get("bias", False)
        args = dict(filter(lambda x: x[0] not in ["method"], block.items()))
        self.activation = activation
        if method in BASEBLOCK:
            self.block = BASEBLOCK[method](
                channels=out_channels,
                activation=activation,
                **args,
            )
        else:
            assert False, f"Unknown block type: {block}"

        self.conv_trans = NormConvTrans1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
            output_padding=0,
            bias=bias,
        )

    def forward(self, x):
        out = self.block(self.activation(self.conv_trans(x)))
        return out


AEBLOCK = {name.lower(): obj for name, obj in globals().items() if inspect.isclass(obj) and issubclass(obj, nn.Module)}
