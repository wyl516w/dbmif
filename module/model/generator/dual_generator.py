import torch
from torch import nn
from ..module import (
    PseudoQMFBanks,
    DeepIterativeAttentionalFeatureFusion,
    EncBlock,
    DecBlock,
    BridgeBlock,
    MultiscaleChannelAttentionModule,
    GateConvFusion,
)


class DualGenerator(nn.Module):

    def __init__(
        self,
        m: int = 4,
        n: int = 32,
        pa: int = 4,
        pb: int = 4,
        channel_list: list = [32, 64, 128, 256],
        stride_list: list = [2, 4, 8],
        block={"method": "residualblock", "num_layers": 3, "dilation": "3**layer", "bias": False},
        bridge={"method": "skip", "fusion": "add"},
    ):
        super(DualGenerator, self).__init__()
        self.pa = pa
        self.pb = pb
        self.multiple = m
        for stride in stride_list:
            self.multiple *= stride
        self.pqmf = PseudoQMFBanks(decimation=m, kernel_size=n)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.first_block = FirstBlock(ac_channels=pa, bc_channels=pb, out_channels=m)
        self.first_fusion = FirstFusion(in_channels=m, out_channels=channel_list[0])
        self.encoder_blocks = nn.ModuleList(
            DualEncBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                stride=stride,
                activation=self.activation,
                block=block,
            )
            for in_ch, out_ch, stride in zip(channel_list[:-1], channel_list[1:], stride_list)
        )
        self.encoder_fusions = nn.ModuleList(
            GateConvFusion(
                in1_channels=ch,
                in2_channels=ch,
                out1_channels=ch,
                out2_channels=ch,
                kernel_size=3,
            )
            for ch in channel_list[1:]
        )
        self.bridges = nn.ModuleList([DualBridgeBlock(method=bridge, channels=channels) for channels in reversed(channel_list[:-1])])
        ratio = self.bridges[0].ratio
        self.latent_block = LatentBlock(
            in_channels=channel_list[-1],
            out_channels=channel_list[-1],
            latent_channels=channel_list[-1] // 4,
            activation=self.activation,
        )
        self.decoder_blocks = nn.ModuleList(
            DualDecBlock(
                in_channels=in_ch * ratio,
                out_channels=out_ch,
                stride=stride,
                activation=self.activation,
                block=block,
            )
            for in_ch, out_ch, stride in zip(reversed(channel_list[1:]), reversed(channel_list[:-1]), reversed(stride_list))
        )
        self.decoder_fusions = nn.ModuleList(
            GateConvFusion(
                in1_channels=ch,
                in2_channels=ch,
                out1_channels=ch,
                out2_channels=ch,
                kernel_size=3,
            )
            for ch in reversed(channel_list[:-1])
        )
        self.last_block = LastBlock(
            in_channels=channel_list[0],
            out_channels=m,
            res_ac=pa,
            res_bc=pb,
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
        f_ac, f_bc = self.first_block(ac_pqmf, bc_pqmf)  # (B, c, T)
        x = self.first_fusion(f_ac, f_bc)
        enc_outputs = []
        for enc, fusion in zip(self.encoder_blocks, self.encoder_fusions):
            x = enc(x)
            enc_outputs.append(x)
            x = fusion(*x)
        x = self.latent_block(x)
        for dec, fusion, bridge, enc_out in zip(self.decoder_blocks, self.decoder_fusions, self.bridges, reversed(enc_outputs)):
            x = bridge(x, enc_out)
            x = dec(x)
            x = fusion(*x)
        # last block
        enhanced_signal_decomposed = self.last_block(x, f_ac, f_bc)
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

    def forward(self, ac, bc):
        return ac, bc


class FirstFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fusion = GateConvFusion(
            in1_channels=in_channels,
            in2_channels=in_channels,
            out1_channels=out_channels,
            out2_channels=out_channels,
            kernel_size=3,
        )

    def forward(self, x1, x2):
        return self.fusion(x1, x2)


class LatentBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels, activation):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=latent_channels,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            activation,
            nn.Conv1d(
                in_channels=latent_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
        )
        self.conv_b = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=latent_channels,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            activation,
            nn.Conv1d(
                in_channels=latent_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
        )

    def forward(self, x):
        a, b = x
        return self.conv_a(a), self.conv_b(b)


class LastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_ac, res_bc):
        super().__init__()
        self.ac_branch = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            MultiscaleChannelAttentionModule(channels=out_channels),
        )
        self.bc_branch = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            MultiscaleChannelAttentionModule(channels=out_channels),
        )
        self.ac_res = nn.Conv1d(
            in_channels=res_ac,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.bc_res = nn.Conv1d(
            in_channels=res_bc,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )
        self.final_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, x, f_ac, f_bc):
        ac, bc = x
        ac = self.ac_branch(ac) * self.ac_res(f_ac)
        bc = self.bc_branch(bc) * self.bc_res(f_bc)
        return self.final_conv(torch.cat([ac, bc], dim=1))


class DualEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, block):
        super().__init__()
        self.enc_a = EncBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            activation=activation,
            block=block,
        )
        self.enc_b = EncBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            activation=activation,
            block=block,
        )

    def forward(self, x):
        a, b = x
        return self.enc_a(a), self.enc_b(b)


class DualDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, block):
        super().__init__()
        self.dec_a = DecBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            activation=activation,
            block=block,
        )
        self.dec_b = DecBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            activation=activation,
            block=block,
        )

    def forward(self, x):
        a, b = x
        return self.dec_a(a), self.dec_b(b)


class DualBridgeBlock(nn.Module):
    def __init__(self, method, channels):
        super().__init__()
        self.bridge_a = BridgeBlock(method=method, channels=channels)
        self.bridge_b = BridgeBlock(method=method, channels=channels)
        self.ratio = self.bridge_a.ratio

    def forward(self, x, res):
        a, b = x
        ra, rb = res
        return self.bridge_a(a, ra), self.bridge_b(b, rb)


if __name__ == "__main__":
    import torch

    ac = torch.randn(4, 1, 16000)
    bc = torch.randn(4, 1, 16000)
    model = DualGenerator()
    y, yd = model(ac, bc)
    print(y.shape)
