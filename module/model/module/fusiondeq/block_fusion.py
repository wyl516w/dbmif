import torch
from torch import nn
import inspect


class DEQFusionBlock(nn.Module):
    def __init__(self, num_out_dims, kernel_size, bias):
        """
        Purified-then-combined fusion block.
        """
        super(DEQFusionBlock, self).__init__()

        self.out_dim = num_out_dims[-1]

        self.gate = nn.Conv1d(num_out_dims[0], self.out_dim, kernel_size=kernel_size, padding="same", bias=bias)
        self.fuse = nn.Conv1d(self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias)

        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.gn3 = nn.GroupNorm(1, self.out_dim, affine=True)

    def forward(self, x, injection_features, residual_feature):
        extracted_feats = []
        for inj_feat in injection_features:
            extracted_feats.append(x * self.gate(inj_feat + x))
        out = self.fuse(torch.stack(extracted_feats, dim=0).sum(dim=0))
        out = out + residual_feature
        out = self.gn3(self.relu3(out))
        return out


FUSIONBLOCKDICT = {name.lower(): obj for name, obj in globals().items() if inspect.isclass(obj) and issubclass(obj, nn.Module)}
