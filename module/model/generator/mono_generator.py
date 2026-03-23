import torch
from torch import nn
from ..module import (
    PseudoQMFBanks,
    DeepIterativeAttentionalFeatureFusion,
    EncBlock,
    DecBlock,
    BridgeBlock,
    MultiscaleChannelAttentionModule,
)


class MonoGenerator(nn.Module):

    def __init__(
        self,
        m: int = 4,
        n: int = 32,
        pa: int = 4,
        pb: int = 4,
        channel_list: list = [32, 64, 128, 256],
        stride_list: list = [2, 4, 8],
        initial_fusion_times=3,
        block={"method": "residualblock", "num_layers": 3, "dilation": "3**layer", "bias": False},
        bridge={"method": "skip", "fusion": "add"},
    ):
        super(MonoGenerator, self).__init__()
        self.pa = pa
        self.pb = pb
        self.multiple = m
        for stride in stride_list:
            self.multiple *= stride
        self.pqmf = PseudoQMFBanks(decimation=m, kernel_size=n)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.first_block = FirstBlock(ac_channels=pa, bc_channels=pb, out_channels=m, initial_times=initial_fusion_times)
        self.first_conv = nn.Conv1d(
            in_channels=m,
            out_channels=channel_list[0],
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.encoder_blocks = nn.ModuleList(
            EncBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                stride=stride,
                activation=self.activation,
                block=block,
            )
            for in_ch, out_ch, stride in zip(channel_list[:-1], channel_list[1:], stride_list)
        )
        self.bridges = nn.ModuleList([BridgeBlock(method=bridge, channels=channels) for channels in reversed(channel_list[:-1])])
        ratio = self.bridges[0].ratio
        self.latent_block = LatentBlock(
            in_channels=channel_list[-1],
            out_channels=channel_list[-1],
            latent_channels=channel_list[-1] // 4,
            activation=self.activation,
        )
        self.decoder_blocks = nn.ModuleList(
            DecBlock(
                in_channels=in_ch * ratio,
                out_channels=out_ch,
                stride=stride,
                activation=self.activation,
                block=block,
            )
            for in_ch, out_ch, stride in zip(reversed(channel_list[1:]), reversed(channel_list[:-1]), reversed(stride_list))
        )
        self.last_block = LastBlock(
            in_channels=channel_list[0],
            out_channels=m,
            res_channels=pa,
        )

    def forward(self, ac, bc):
        """
        Args:
            ac (torch.Tensor): acoustic features (B, pa, T)
            bc (torch.Tensor): bottleneck features (B, pb, T)
        Returns:
            y (torch.Tensor): generated waveform (B, 1, T * m)
        """
        ac, pad_len = self.pad_tensor(ac)
        bc, pad_len = self.pad_tensor(bc)
        ac_pqmf = self.pqmf(ac, "analysis", bands=self.pa)
        bc_pqmf = self.pqmf(bc, "analysis", bands=self.pb)
        f_pmqf = self.first_block(ac_pqmf, bc_pqmf)  # (B, c, T)
        x = self.first_conv(f_pmqf)
        enc_outputs = []
        for enc in self.encoder_blocks:
            x = enc(x)
            enc_outputs.append(x)
        x = self.latent_block(x)
        for dec, bridge, res in zip(self.decoder_blocks, self.bridges, reversed(enc_outputs)):
            x = bridge(x, res)
            x = dec(x)

        # last block
        enhanced_signal_decomposed = self.last_block(x, f_pmqf)
        enhanced_signal = torch.sum(self.pqmf(enhanced_signal_decomposed, "synthesis"), 1, keepdim=True)
        self.external_loss_value = 0
        enhanced_signal = self.restore_tensor(enhanced_signal, pad_len)
        return enhanced_signal, enhanced_signal_decomposed

    def pqmf_analysis(self, signal):
        """This function is used to perform PQMF analysis"""
        signal, _ = self.pad_tensor(signal)
        return self.pqmf(signal, "analysis")

    def pad_tensor(self, tensor):
        """This function is used to make tensor's dim 2 len divisible by multiple"""
        old_len = tensor.shape[2]
        pad_len = self.multiple - (old_len + self.pqmf.kernel_size) % self.multiple
        tensor = torch.nn.functional.pad(tensor, (0, pad_len), "constant", 0)
        return tensor, pad_len

    def restore_tensor(self, tensor, pad_len):
        """This function is used to restore tensor's dim 2 len"""
        new_len = tensor.shape[2] - pad_len
        return tensor[:, :, :new_len]

    def cut_tensor(self, tensor):
        """This function is used to make tensor's dim 2 len divisible by multiple"""

        old_len = tensor.shape[2]
        new_len = old_len - (old_len + self.pqmf.kernel_size) % self.multiple
        tensor = torch.narrow(tensor, 2, 0, new_len)

        return tensor


class FirstBlock(nn.Module):

    def __init__(self, ac_channels, bc_channels, out_channels, initial_times=3):
        super().__init__()
        self.first_conv_ac = nn.Conv1d(
            in_channels=ac_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.first_conv_bc = nn.Conv1d(
            in_channels=bc_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.first_conv_fusion = DeepIterativeAttentionalFeatureFusion(
            channels=out_channels,
            repeat_times=initial_times,
        )

    def forward(self, ac, bc):
        return self.first_conv_fusion(self.first_conv_ac(ac), self.first_conv_bc(bc))


class LatentBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, activation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=latent_channels,
            kernel_size=7,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.conv2 = nn.Conv1d(
            in_channels=latent_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.activation = activation

    def forward(self, x):
        return self.conv2(self.activation(self.conv1(x)))


class LastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.mscam = MultiscaleChannelAttentionModule(channels=out_channels)
        self.res = nn.Conv1d(
            in_channels=res_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, x, res):
        return self.mscam(self.conv(x)) * self.res(res)
