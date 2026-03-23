"""DBMIF generator and sub blocks definition in Pytorch"""

import torch
from torch import nn
from ..module import (
    PseudoQMFBanks,
    DEQEQFusionBlock,
    BridgeBlock,
    MultiscaleChannelAttentionModule,
    AttentionalFeatureFusion,
    IterativeAttentionalFeatureFusion,
    DeepIterativeAttentionalFeatureFusion,
    EncBlock,
    DecBlock,
    BridgeBlock,
    GateConvFusion,
    GateConvSideFusion,
    ConvFusion,
)
from ..module.block.unit import LSTM


class Generator(nn.Module):
    """
    Generator Network Module
    """

    def __init__(
        self,
        m: int = 4,
        n: int = 32,
        pa: int = 4,
        pb: int = 1,
        channel_list: list = [32, 64, 128, 256],
        stride_list: list = [2, 4, 8],
        block={"method": "residual", "num_layers": 3, "dilation": "3**layer", "bias": False},
        encoder_block=None,
        decoder_block=None,
        fusion={"method": "conv"},
        initial_times=3,
        first_fusion=None,
        encoder_fusion=None,
        latent_fusion=None,
        decoder_fusion=None,
        bridge={"method": "skip", "fusion": "add"},
        last_bridge_ac={"method": "skip", "fusion": "add"},
        last_bridge_bc={"method": "skip", "fusion": "add"},
        last_bridge_fusion={"method": "skip", "fusion": "add"},
    ):
        """
        Generator of DBMIF
        Args:
            m:              The number of PQMF bands, which is also the decimation factor of the waveform after the analysis step
            n:              The kernel size of PQMF
            pa:             The number of informative PMQF bands sent to the generator for ac component
            pb:             The number of informative PMQF bands sent to the generator for bc component
            channel_list:   The list of channel in each layer
            stride_list:    The list of stride in each layer
            bridge:         The bridge method of generator, which can be "skip" or "pconv",
                            and the fusion method can be "add" or "concat"
            block:          The encoder/decoder block method of generator, which can be "residual" or "denseconv",
                            and the number of layers must be positive,
                            and the dilation is an expression of layer(begin from 0)
            fusion:         The fusion method of generator, which can be "conv" or "auto_deep"
            lstm:           The LSTM layer in latent convolution, which can be "bidirectional" , True or None
        """
        super().__init__()
        assert len(channel_list) == len(stride_list) + 1, f"Length of channel_list must be len(stride_list) + 1"

        for channels in channel_list:
            assert channels % 2 == 0, f"The number of channels must be even, got {channels} in {channel_list}"
        block = {} if block is None else block
        fusion = {} if fusion is None else fusion
        bridge = {} if bridge is None else bridge
        encoder_block = block if encoder_block is None else encoder_block
        decoder_block = block if decoder_block is None else decoder_block
        first_fusion = fusion if first_fusion is None else first_fusion
        encoder_fusion = fusion if encoder_fusion is None else encoder_fusion
        latent_fusion = fusion if latent_fusion is None else latent_fusion
        decoder_fusion = fusion if decoder_fusion is None else decoder_fusion

        self.pa = pa
        self.pb = pb
        self.pqmf = PseudoQMFBanks(decimation=m, kernel_size=n)
        self.multiple = m
        for stride in stride_list:
            self.multiple *= stride
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
        self.first_block = TripleFirstBlock(
            ac_channels=pa,
            bc_channels=pb,
            out_channels=m,
            kernel_size=3,
            bias=False,
            initial_times=initial_times,
        )
        self.first_fusion = TripleFusionBlock(
            in_channels=m,
            out_channels=channel_list[0],
            kernel_size=3,
            fusion=first_fusion,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TripleEncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=self.activation,
                    block=encoder_block,
                )
                for in_channels, out_channels, stride in zip(channel_list[:-1], channel_list[1:], stride_list)
            ]
        )
        self.encoder_fusions = nn.ModuleList(
            [
                TripleFusionBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    fusion=encoder_fusion,
                )
                for channels in channel_list[1:]
            ]
        )
        self.latent_block = TripleLatentConv(
            in_channels=channel_list[-1],
            out_channels=channel_list[-1],
            latent_channels=channel_list[-1] // 4,
            activation=self.activation,
            fusion=latent_fusion,
        )
        self.bridge_fusions = nn.ModuleList([TripleBridgeBlock(method=bridge, channels=channels, bias=False) for channels in channel_list[-1::-1]])
        ratio = self.bridge_fusions[0].ratio
        self.decoder_blocks = nn.ModuleList(
            [
                TripleDecoderBlock(
                    in_channels=in_channels * ratio,
                    out_channels=out_channels,
                    stride=stride,
                    activation=self.activation,
                    block=decoder_block,
                )
                for in_channels, out_channels, stride in zip(channel_list[-1::-1], channel_list[-2::-1], stride_list[-1::-1])
            ]
        )
        self.decoder_fusions = nn.ModuleList(
            [
                TripleFusionBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=1,
                    fusion=decoder_fusion,
                )
                for channels in channel_list[-2::-1]
            ]
        )
        self.last_block = TripleLastBlock(
            in_channels=channel_list[0],
            out_channels=m,
            kernel_size=1,
            bias=False,
            ac_res_channels=pa,
            bc_res_channels=pb,
            ac_method=last_bridge_ac,
            bc_method=last_bridge_bc,
            fusion_method=last_bridge_fusion,
        )

    def forward(self, ac, bc):
        ac, pad_len = self.pad_tensor(ac)
        bc, pad_len = self.pad_tensor(bc)
        ac_pqmf = self.pqmf(ac, "analysis", bands=self.pa)
        bc_pqmf = self.pqmf(bc, "analysis", bands=self.pb)
        first_block = self.first_block(ac_pqmf, bc_pqmf)
        inputs = self.first_fusion(first_block)
        bridges = []
        for i in range(len(self.encoder_blocks)):
            inputs = self.encoder_blocks[i](inputs)
            bridges.append(inputs)
            inputs = self.encoder_fusions[i](inputs)
        inputs = self.latent_block(inputs)
        if hasattr(self.latent_block, "external_loss_value"):
            self.external_loss_value = self.latent_block.external_loss_value
        for i in range(len(self.decoder_blocks)):
            inputs = self.bridge_fusions[i](bridges.pop(), inputs)
            inputs = self.decoder_blocks[i](inputs)
            inputs = self.decoder_fusions[i](inputs)
        enhanced_signal_decomposed = self.last_block(inputs, first_block)
        enhanced_signal_decomposed = self.tanh(enhanced_signal_decomposed)
        enhanced_signal = torch.sum(self.pqmf(enhanced_signal_decomposed, "synthesis"), 1, keepdim=True)
        enhanced_signal = self.restore_tensor(enhanced_signal, pad_len)
        return enhanced_signal, enhanced_signal_decomposed

    @property
    def external_loss(self):
        assert hasattr(self, "external_loss_value"), "No external loss value, Run forward first."
        loss = self.external_loss_value
        return loss

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


class TripleFirstBlock(nn.Module):
    """
    Triple First Convolution Module
    """

    def __init__(self, ac_channels, bc_channels, out_channels, kernel_size, bias=False, initial_times=3):
        super().__init__()
        if ac_channels != out_channels:
            self.first_conv_ac = nn.Conv1d(
                in_channels=ac_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=bias,
                padding_mode="reflect",
            )
        else:
            self.first_conv_ac = nn.Identity()
        if bc_channels != out_channels:
            self.first_conv_bc = nn.Conv1d(
                in_channels=bc_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=bias,
                padding_mode="reflect",
            )
        else:
            self.first_conv_bc = nn.Identity()
        if initial_times == 0:
            def first_conv_fusion(ac, bc):
                return ac + bc
            self.first_conv_fusion = first_conv_fusion
        elif initial_times == 1:
            self.first_conv_fusion = AttentionalFeatureFusion(out_channels)
        elif initial_times == 2:
            self.first_conv_fusion = IterativeAttentionalFeatureFusion(out_channels)
        elif initial_times == 3:
            self.first_conv_fusion = DeepIterativeAttentionalFeatureFusion(out_channels)
        else:
            assert False, f"Unknown initial_times: {initial_times}"

    def forward(self, ac, bc):
        new_ac = self.first_conv_ac(ac)
        new_bc = self.first_conv_bc(bc)
        new_fusion = self.first_conv_fusion(new_ac, new_bc)
        return new_ac, new_bc, new_fusion


class TripleLastBlock(nn.Module):
    """
    Triple Last Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        ac_res_channels=None,
        bc_res_channels=None,
        ac_method={"method": "skip", "fusion": "add"},
        bc_method={"method": "skip", "fusion": "add"},
        fusion_method={"method": "skip", "fusion": "add"},
    ):
        super().__init__()
        self.mscam = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding="same",
                        bias=bias,
                        padding_mode="reflect",
                    ),
                    MultiscaleChannelAttentionModule(channels=out_channels, inter_channels=out_channels // 2),
                )
                for _ in range(3)
            ]
        )
        self.res = nn.ModuleList()
        self.bridge = nn.ModuleList()
        channels = 3 * out_channels
        if ac_res_channels and ac_method.get("method", "add") != "none":
            ac_res = nn.Conv1d(ac_res_channels, out_channels, 1)
            ac_bridge = BridgeBlock(ac_method, out_channels, bias)
            self.res.append(ac_res)
            self.bridge.append(ac_bridge)
            channels += (ac_bridge.channel_ratio - 1) * out_channels
        else:
            self.res.append(nn.Identity())
            self.bridge.append(BridgeBlock({"method": "skip", "fusion": "none"}, out_channels, bias))
        if bc_res_channels and bc_method.get("method", "add") != "none":
            bc_res = nn.Conv1d(bc_res_channels, out_channels, 1)
            bc_bridge = BridgeBlock(bc_method, out_channels, bias)
            self.res.append(bc_res)
            self.bridge.append(bc_bridge)
            channels += (bc_bridge.channel_ratio - 1) * out_channels
        else:
            self.res.append(nn.Identity())
            self.bridge.append(BridgeBlock({"method": "skip", "fusion": "none"}, out_channels, bias))
        if fusion_method.get("method", "add") != "none":
            fusion_res = nn.Conv1d(out_channels, out_channels, 1)
            fusion_bridge = BridgeBlock(fusion_method, out_channels, bias)
            self.res.append(fusion_res)
            self.bridge.append(fusion_bridge)
            channels += (fusion_bridge.channel_ratio - 1) * out_channels
        else:
            self.res.append(nn.Identity())
            self.bridge.append(BridgeBlock({"method": "skip", "fusion": "none"}, out_channels, bias))
        self.last_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, inputs, residuals=None):
        inputs = [mscam(model_input) for mscam, model_input in zip(self.mscam, inputs)]
        inputs = [bridge(res(residual), model_input) for res, bridge, model_input, residual in zip(self.res, self.bridge, inputs, residuals)]
        abf = self.last_conv(torch.cat(inputs, dim=1))
        return abf


class TripleBridgeBlock(nn.Module):
    """
    Triple Bridge Module
    """

    def __init__(self, method={"method": "skip", "fusion": "add"}, channels=None, bias=False):
        super().__init__()
        self.bridge = nn.ModuleList([BridgeBlock(method, channels, bias) for _ in range(3)])

    def forward(self, encoders, decoders):
        return [bridge(encoder, decoder) for bridge, encoder, decoder in zip(self.bridge, encoders, decoders)]

    @property
    def ratio(self):
        return self.bridge[0].ratio


class TripleFusionBlock(nn.Module):
    """
    Triple Fusion Module
    """

    def __init__(self, in_channels, out_channels, kernel_size, fusion={}):
        super().__init__()
        method = fusion.get("method", "gateconv")
        args = dict(filter(lambda x: x[0] not in ["method"], fusion.items()))
        if method == "gateconv":
            self.fusion = TripleGateConvFusionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                **args,
            )
        elif method == "auto":
            self.fusion = TripleAutoFusionBlock(
                in_channels=in_channels,
                mid_channels=in_channels // 4,
                out_channels=out_channels,
                kernel_size=kernel_size,
                **args,
            )
        elif method == "conv":
            self.fusion = TripleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                **args,
            )
        elif method == "none":
            self.fusion = TripleIdentityBlock(**args)
        else:
            assert False, f"Unknown fusion method: {method}"

    def forward(self, inputs):
        outputs = self.fusion(inputs)
        if hasattr(self.fusion, "external_loss_value"):
            self.external_loss_value = self.fusion.external_loss_value
        return outputs


class TripleIdentityBlock(nn.Module):
    """
    Triple Identity Module
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class TripleLSTMBlock(nn.Module):
    """
    Triple LSTM Module
    """

    def __init__(self, channels, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.ModuleList(
            [
                LSTM(
                    channels=channels,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                )
                for _ in range(3)
            ]
        )

    def forward(self, inputs):
        return [lstm(model_input) for lstm, model_input in zip(self.lstm, inputs)]


# class TripleGateConvFusionBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         bias=False,
#         residual=False,
#     ):
#         super().__init__()
#         self.gated_conv_ac = GateConvFusion(
#             in1_channels=in_channels,
#             in2_channels=in_channels,
#             out1_channels=out_channels,
#             out2_channels=out_channels,
#             kernel_size=kernel_size,
#             bias=bias,
#             residual=residual,
#             activation=nn.Tanh(),
#         )
#         self.gated_conv_bc = GateConvFusion(
#             in1_channels=in_channels,
#             in2_channels=in_channels,
#             out1_channels=out_channels,
#             out2_channels=out_channels,
#             kernel_size=kernel_size,
#             bias=bias,
#             residual=residual,
#             activation=nn.Tanh(),
#         )

#     def forward(self, inputs):
#         ac, bc, fusion = inputs
#         new_ac, new_fusion_ac = self.gated_conv_ac(ac, fusion)
#         new_bc, new_fusion_bc = self.gated_conv_bc(bc, fusion)
#         new_fusion = new_fusion_ac + new_fusion_bc
#         return new_ac, new_bc, new_fusion

class TripleGateConvFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        residual=False,
    ):
        super().__init__()
        
        self.gated_conv_ac = GateConvFusion(
            in1_channels=in_channels,
            in2_channels=in_channels,
            out1_channels=out_channels,
            out2_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            residual=residual,
            activation=nn.Tanh(),
        )
        self.gated_conv_bc = GateConvFusion(
            in1_channels=in_channels,
            in2_channels=in_channels,
            out1_channels=out_channels,
            out2_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            residual=residual,
            activation=nn.Tanh(),
        )

    def forward(self, inputs):
        ac, bc, fusion = inputs
        new_ac, new_fusion_ac = self.gated_conv_ac(ac, fusion)
        new_bc, new_fusion_bc = self.gated_conv_bc(bc, fusion)
        new_fusion = new_fusion_ac + new_fusion_bc
        return new_ac, new_bc, new_fusion


class TripleAutoFusionBlock(nn.Module):
    """
    Triple Gated Convolution Module with auto-fit deep
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        bias=False,
        residual=False,
        **kwargs,
    ):
        super().__init__()
        self.front_block = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=1,
                    padding="same",
                    bias=bias,
                    padding_mode="reflect",
                )
                for _ in range(3)
            ]
        )
        self.fusion_block = DEQEQFusionBlock([mid_channels, mid_channels, mid_channels], kernel_size, bias, **kwargs)
        self.back_block = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    bias=bias,
                    padding_mode="reflect",
                )
                for _ in range(3)
            ]
        )
        self.residual = residual
        self.loss = 0

    def forward(self, inputs):
        mid_inputs = [frontblock(model_input) for frontblock, model_input in zip(self.front_block, inputs)]
        mid_outputs, jacbian_loss, _ = self.fusion_block(mid_inputs)
        outputs = [backblock(model_output) for backblock, model_output in zip(self.back_block, mid_outputs)]
        if self.residual:
            outputs = [model_input + model_output for model_input, model_output in zip(inputs, outputs)]
        self.external_loss_value = jacbian_loss.mean().item()
        return outputs


class TripleConvBlock(nn.Module):
    """
    Triple Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        residual=False,
    ):
        super().__init__()

        self.gated_conv_ac = ConvFusion(
            in1_channels=in_channels,
            in2_channels=in_channels,
            out1_channels=out_channels,
            out2_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            residual=residual,
            activation=nn.Tanh(),
        )
        self.gated_conv_bc = ConvFusion(
            in1_channels=in_channels,
            in2_channels=in_channels,
            out1_channels=out_channels,
            out2_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            residual=residual,
            activation=nn.Tanh(),
        )

    def forward(self, inputs):
        ac, bc, fusion = inputs
        new_ac, new_fusion_ac = self.gated_conv_ac(ac, fusion)
        new_bc, new_fusion_bc = self.gated_conv_bc(bc, fusion)
        new_fusion = new_fusion_ac + new_fusion_bc
        return new_ac, new_bc, new_fusion


class TripleEncoderBlock(nn.Module):
    """
    Triple Encoder Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        activation,
        block={
            "method": "residual",
            "num_layers": 3,
            "dilation": "3**layer",
            "bias": False,
        },
    ):
        super().__init__()
        self.encoder = nn.ModuleList([EncBlock(in_channels, out_channels, stride, activation, block) for _ in range(3)])

    def forward(self, inputs):
        return [encoder(model_input) for encoder, model_input in zip(self.encoder, inputs)]


class TripleDecoderBlock(nn.Module):
    """
    Triple Decoder Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        activation,
        block={
            "method": "residual",
            "num_layers": 3,
            "dilation": "3**layer",
            "bias": False,
        },
    ):
        super().__init__()
        self.decoder = nn.ModuleList([DecBlock(in_channels, out_channels, stride, activation, block) for _ in range(3)])

    def forward(self, inputs):
        return [decoder(model_input) for decoder, model_input in zip(self.decoder, inputs)]


class TripleLatentConv(nn.Module):
    """
    Triple Latent Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        activation,
        fusion={"method": "none", "bias": False},
    ):
        super().__init__()
        bias = fusion.get("bias", False)
        self.preprocess = nn.ModuleList(
            [
                nn.Sequential(
                    activation,
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=latent_channels,
                        kernel_size=7,
                        padding="same",
                        bias=bias,
                        padding_mode="reflect",
                    ),
                    activation,
                )
                for _ in range(3)
            ]
        )
        self.latent = TripleFusionBlock(
            in_channels=latent_channels,
            out_channels=latent_channels,
            kernel_size=3,
            fusion=fusion,
        )
        self.postprocess = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=latent_channels,
                        out_channels=out_channels,
                        kernel_size=7,
                        padding="same",
                        bias=bias,
                        padding_mode="reflect",
                    ),
                    activation,
                )
                for _ in range(3)
            ]
        )

    def forward(self, inputs):
        mid_inputs = [preprocess(model_input) for preprocess, model_input in zip(self.preprocess, inputs)]
        mid_outputs = self.latent(mid_inputs)
        if hasattr(self.latent, "external_loss_value"):
            self.external_loss_value = self.latent.external_loss_value
        return [postprocess(model_input) for postprocess, model_input in zip(self.postprocess, mid_outputs)]


if __name__ == "__main__":
    # Instantiate nn.module
    generator = Generator(
        m=4,
        n=32,
        pa=4,
        pb=1,
        channel_list=[32, 64, 128, 256],
        stride_list=[2, 4, 8],
        bridges={"method": "pconv", "fusion": "concat"},
        block={"method": "denseconv", "num_layers": 5, "dilation": "2*layer+1"},
        fusion={"method": "auto_deep"},
        lstm=True,
    )

    # Instantiate tensor with shape: (batch_size, channel, time_len)
    ac = torch.randn((5, 1, 16000))
    bc = torch.randn((5, 1, 16000))
    #  cut tensor to ensure forward pass run well
    ac, bc = generator.cut_tensor(ac), generator.cut_tensor(bc)

    # Test forward of model
    enhanced_signal, enhanced_signal_decomposed = generator(ac, bc)
    print(
        f"enhanced_signal.shape: {enhanced_signal.shape}",
        f"enhanced_signal_decomposed.shape: {enhanced_signal_decomposed.shape}",
    )
    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in generator.parameters())
    print(f"pytorch_total_params: {pytorch_total_params * 1e-6:.2f} Millions")
