import torch
from torch import nn
from .ae import EncBlock, DecBlock
from .bridge import BridgeBlock


class ParallelEncBlock(nn.Module):
    def __init__(
        self,
        in_channels=[64, 64, 64],
        out_channels=[64, 64, 64],
        stride=1,
        activation=nn.LeakyReLU(0.001),
        block={
            "method": "residualblock",
            "num_layers": 3,
            "dilation": "3**layer",
            "residual": False,
            "bias": False,
        },
    ):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [
                EncBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    stride=stride,
                    activation=activation,
                    block=block,
                )
                for i in range(len(in_channels))
            ]
        )

    def forward(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)) or not all(isinstance(i, torch.Tensor) for i in x):
            raise ValueError("Input should be a list or tuple of tensors.")
        if len(x) != len(self.enc_blocks):
            raise ValueError("The number of inputs should be equal to the number of encoder blocks.")
        outputs = [block(x[i]) for i, block in enumerate(self.enc_blocks)]
        return tuple(outputs)


class ParallelDecBlock(nn.Module):
    def __init__(
        self,
        in_channels=[64, 64, 64],
        out_channels=[64, 64, 64],
        stride=1,
        activation=nn.LeakyReLU(0.001),
        block={
            "method": "residualblock",
            "num_layers": 3,
            "dilation": "3**layer",
            "residual": False,
            "bias": False,
        },
        bridge={
            "method": "skip",
            "fusion": "none",
        },
    ):
        super().__init__()
        self.dec_blocks = nn.ModuleList(
            [
                DecBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    stride=stride,
                    activation=activation,
                    block=block,
                )
                for i in range(len(in_channels))
            ]
        )
        self.bridge_blocks = nn.ModuleList(
            [
                BridgeBlock(
                    method=bridge,
                    channels=in_channels[i],
                )
                for i in range(len(in_channels))
            ]
        )

    def forward(self, x, bridge=None):
        if not (isinstance(x, list) or isinstance(x, tuple)) or not all(isinstance(i, torch.Tensor) for i in x):
            raise ValueError("Input should be a list or tuple of tensors.")
        if len(x) != len(self.dec_blocks):
            raise ValueError("The number of inputs should be equal to the number of decoder blocks.")
        if bridge is not None:
            if not (isinstance(bridge, list) or isinstance(bridge, tuple)) or not all(isinstance(i, torch.Tensor) for i in bridge):
                raise ValueError("Bridge input should be a list or tuple of tensors, otherwise you should ignored this paras.")
            if len(bridge) != len(self.bridge_blocks):
                raise ValueError("The number of bridge inputs should be equal to the number of bridge blocks.")
            x = [block(bridge[i], x[i]) for i, block in enumerate(self.bridge_blocks)]
        outputs = [block(x[i]) for i, block in enumerate(self.dec_blocks)]
        return tuple(outputs)


class ParallelAnyBlock(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        # check if args or kwargs has list, if has, then split for various inputs
        self.num_model = 1
        for arg in args:
            if isinstance(arg, list):
                self.num_model = len(arg)
                break
        for key, value in kwargs.items():
            if isinstance(value, list):
                self.num_model = len(value)
                break
        self.blocks = nn.ModuleList()
        for i in range(self.num_model):
            self.blocks.append(
                model(
                    *[arg[i] if isinstance(arg, list) else arg for arg in args],
                    **{key: value[i] if isinstance(value, list) else value for key, value in kwargs.items()},
                )
            )

    def forward(self, *args, **kwargs):
        for arg in args:
            if not (isinstance(arg, list) or isinstance(arg, tuple)) or not all(isinstance(i, torch.Tensor) for i in arg):
                raise ValueError("Input should be a list or tuple of tensors.")
            if len(arg) != self.num_model:
                raise ValueError("The number of inputs should be equal to the number of decoder blocks.")
        for key, value in kwargs.items():
            if not (isinstance(value, list) or isinstance(value, tuple)) or not all(isinstance(i, torch.Tensor) for i in value):
                raise ValueError("Input should be a list or tuple of tensors.")
            if len(value) != self.num_model:
                raise ValueError("The number of inputs should be equal to the number of decoder blocks.")
        outputs = [
            block(*[arg[i] if isinstance(arg, list) else arg for arg in args], **{key: value[i] if isinstance(value, list) else value for key, value in kwargs.items()})
            for i, block in enumerate(self.blocks)
        ]
        return tuple(outputs)

