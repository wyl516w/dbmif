import torch
from torch import nn
from .GateConvFusion import GateConvFusion
from .ConvAttentionFusion import ConvattFusion
FUSEDICT={
    "gateconvfusion": GateConvFusion,
    "convattfusion": ConvattFusion,
}

class ParallelFusion(nn.Module):
    def __init__(
        self,
        in_channels=[64, 64, 64],
        out_channels=[64, 64, 64],
        kernel_size=3,
        fusion={"method": "gateconvfusion"},
    ):
        super().__init__()
        method = fusion.get("method", "gateconvfusion")
        model = FUSEDICT.get(method, FUSEDICT.get("gateconvfusion"))
        args = dict(filter(lambda x: x[0] not in ["method"], fusion.items()))
        self.fusion_dict = nn.ModuleList(
            [
                model(
                    in1_channels=in_channels[i],
                    in2_channels=in_channels[-1],
                    out1_channels=out_channels[i],
                    out2_channels=out_channels[-1],
                    interaction_kernel_size=kernel_size,
                    **args,
                )
                for i in range(len(in_channels) - 1)
            ]
        )

    def forward(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)) or not all(isinstance(i, torch.Tensor) for i in x):
            raise ValueError("Input should be a list or tuple of tensors.")
        if len(x) != len(self.fusion_dict) + 1:
            raise ValueError("The number of inputs should be equal to the number of blocks.")
        modalitys = x[:-1]
        fusion = x[-1]
        outputs = []
        output_fusion = 0
        for i in range(len(self.fusion_dict)):
            out_modality, out_fusion = self.fusion_dict[i](modalitys[i], fusion)
            outputs.append(out_modality)
            output_fusion += out_fusion
        outputs.append(output_fusion)
        return tuple(outputs)
