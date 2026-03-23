"""EBEN-compatible generator implementation."""

import torch
from torch import nn

from ..module import PseudoQMFBanks


class GeneratorDBMIF(nn.Module):
    def __init__(
        self,
        m: int,
        n: int,
        p: int = 1,
        scale_factor: int = 1,
        is_bridge: bool = True,
    ):
        super().__init__()

        self.p = p
        self.pqmf = PseudoQMFBanks(decimation=m, kernel_size=n)
        self.multiple = 2 * 4 * 8 * m
        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = nn.Conv1d(
            in_channels=p,
            out_channels=32 * scale_factor,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncBlock(out_channels=64 * scale_factor, stride=2, nl=self.nl),
                EncBlock(out_channels=128 * scale_factor, stride=4, nl=self.nl),
                EncBlock(out_channels=256 * scale_factor, stride=8, nl=self.nl),
            ]
        )

        self.latent_conv = nn.Sequential(
            self.nl,
            normalized_conv1d(
                in_channels=256 * scale_factor,
                out_channels=64 * scale_factor,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            self.nl,
            normalized_conv1d(
                in_channels=64 * scale_factor,
                out_channels=256 * scale_factor,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            self.nl,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecBlock(out_channels=128 * scale_factor, stride=8, nl=self.nl),
                DecBlock(out_channels=64 * scale_factor, stride=4, nl=self.nl),
                DecBlock(out_channels=32 * scale_factor, stride=2, nl=self.nl),
            ]
        )

        self.last_conv = nn.Conv1d(
            in_channels=32 * scale_factor,
            out_channels=m,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.is_bridge = is_bridge

    def forward(self, cut_audio):
        cut_audio, pad_len = self.pad_tensor(cut_audio)
        first_bands = self.pqmf(cut_audio, "analysis", bands=self.p)

        x = self.first_conv(first_bands)
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))
        x = self.latent_conv(x3)

        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)
        x = self.last_conv(x)

        if self.is_bridge:
            batch_size, _, time_len = first_bands.shape
            fill_up_tensor = torch.zeros(
                (batch_size, self.pqmf.decimation - self.p, time_len),
                device=first_bands.device,
                dtype=first_bands.dtype,
            )
            cat_tensor = torch.cat(tensors=(first_bands, fill_up_tensor), dim=1)
            enhanced_speech_decomposed = torch.tanh(x + cat_tensor)
        else:
            enhanced_speech_decomposed = torch.tanh(x)

        enhanced_speech = torch.sum(
            self.pqmf(enhanced_speech_decomposed, "synthesis"), 1, keepdim=True
        )
        enhanced_speech = self.restore_tensor(enhanced_speech, pad_len)
        return enhanced_speech, enhanced_speech_decomposed

    def pad_tensor(self, tensor):
        old_len = tensor.shape[2]
        pad_len = self.multiple - (old_len + self.pqmf.kernel_size) % self.multiple
        tensor = torch.nn.functional.pad(tensor, (0, pad_len), "constant", 0)
        return tensor, pad_len

    def restore_tensor(self, tensor, pad_len):
        new_len = tensor.shape[2] - pad_len
        return tensor[:, :, :new_len]

    def cut_tensor(self, tensor):
        old_len = tensor.shape[2]
        new_len = old_len - (old_len + self.pqmf.kernel_size) % self.multiple
        return torch.narrow(tensor, 2, 0, new_len)


class DecBlock(nn.Module):
    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.nl = nl
        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels, nl=nl, dilation=9),
        )
        self.conv_trans = normalized_conv_trans1d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
            output_padding=0,
            bias=bias,
        )

    def forward(self, x, encoder_output):
        x = x + encoder_output
        return self.residuals(self.nl(self.conv_trans(x)))


class EncBlock(nn.Module):
    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=9),
        )
        self.conv = normalized_conv1d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride - 1,
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, x):
        return self.conv(self.residuals(x))


class ResidualUnit(nn.Module):
    def __init__(self, channels, nl, dilation, bias=False):
        super().__init__()

        self.dilated_conv = normalized_conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.pointwise_conv = normalized_conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.nl = nl

    def forward(self, x):
        return x + self.nl(self.pointwise_conv(self.dilated_conv(x)))


def normalized_conv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def normalized_conv_trans1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


GeneratorEBEN = GeneratorDBMIF
