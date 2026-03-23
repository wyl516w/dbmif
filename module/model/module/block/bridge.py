import torch
from torch import nn
from torch.nn import functional as F
from ..fusion.ConvAttentionFusion import ConvAttention


class BridgeBlock(nn.Module):
    """
    Bridge Block Module
    """

    def __init__(self, method={"method": "skip", "fusion": "add"}, channels=None, bias=False):
        super().__init__()
        bridge_method = method.get("method", "skip")
        bridge_fusion = method.get("fusion", "add")
        if bridge_method == "skip":
            self.bridge = nn.Identity()
        elif bridge_method == "pconv":
            assert channels is not None, "channels must be provided for Channel Convolution"
            self.bridge = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                bias=bias,
            )
        else:
            assert False, f"Unknown bridge method: {bridge_method}"

        if bridge_fusion == "add":
            self.bridge_fusion = lambda encoder, decoder: decoder + encoder
            self.channel_ratio = 1

        elif bridge_fusion == "concat":
            self.bridge_fusion = lambda encoder, decoder: torch.cat([decoder, encoder], dim=1)
            self.channel_ratio = 2

        elif bridge_fusion == "pointwise":
            self.bridge_fusion = lambda encoder, decoder: decoder * encoder
            self.channel_ratio = 1

        elif bridge_fusion == "gated":
            self.bridge_fusion = lambda encoder, decoder: decoder * torch.sigmoid(encoder)
            self.channel_ratio = 1

        elif bridge_fusion == "gateatt":
            assert channels is not None, "channels must be provided for Attentinal Gated"
            self.conv_local = nn.Conv1d(channels, channels, kernel_size=1)
            self.conv_global = nn.Conv1d(1, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
            self.pwconv = nn.Conv1d(channels, channels, kernel_size=1)

            def fusion(encoder, decoder):
                inputs = encoder + decoder
                local_att = self.conv_local(inputs)
                global_mean = torch.mean(inputs, dim=1, keepdim=True)
                global_att = self.conv_global(global_mean)
                attn_coeff = self.sigmoid(local_att + global_att)
                encoder_attn = attn_coeff * encoder
                encoder_attn = self.pwconv(encoder_attn)
                return torch.cat([decoder, encoder_attn], dim=1)

            self.bridge_fusion = fusion
            self.channel_ratio = 2
        elif bridge_fusion == "gateattadd":
            assert channels is not None, "channels must be provided for Attentinal Gated Add"
            self.conv_local = nn.Conv1d(channels, channels, kernel_size=1)
            self.conv_global = nn.Conv1d(1, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
            self.pwconv = nn.Conv1d(channels, channels, kernel_size=1)

            def fusion(encoder, decoder):
                inputs = encoder + decoder
                local_att = self.conv_local(inputs)
                global_mean = torch.mean(inputs, dim=1, keepdim=True)
                global_att = self.conv_global(global_mean)
                attn_coeff = self.sigmoid(local_att + global_att)
                encoder_attn = attn_coeff * encoder
                encoder_attn = self.pwconv(encoder_attn)
                return decoder + encoder_attn

            self.bridge_fusion = fusion
            self.channel_ratio = 1
        elif bridge_fusion == "convatt":
            assert channels is not None, "channels must be provided for Convolutional Attention"
            self.convatt = ConvAttention(
                in1_channels=channels,
                in2_channels=channels,
                out_channels=channels,
                interaction_kernel_size=3,
                inter_channels=channels // 2,
                bias=bias,
                residual=False,
                activation=nn.Tanh(),
            )
            self.bridge_fusion = self.convatt
            self.channel_ratio = 1
        elif bridge_fusion == "none":
            self.bridge_fusion = lambda encoder, decoder: decoder
            self.channel_ratio = 1

        else:
            assert False, f"Unknown bridge fusion method: {bridge_fusion}"

    def forward(self, encoder, decoder):
        resolved_encoder = self.bridge(encoder)
        # if decoder.shape[2] > encoder.shape[2]:
        #     decoder = decoder[:, :, : encoder.shape[2]]
        # elif decoder.shape[2] < encoder.shape[2]:
        #     decoder = F.pad(decoder, (0, encoder.shape[2] - decoder.shape[2]))
        resolved_decoder = self.bridge_fusion(resolved_encoder, decoder)
        return resolved_decoder

    @property
    def ratio(self):
        return self.channel_ratio
