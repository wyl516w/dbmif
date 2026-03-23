from torch import nn
from ..block.quaternion1d import QuaternionConv1d, QuaternionBatchNorm1d
import inspect


class QuaternionChannelAttentionModule(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = int(channels**0.5)
        self.local_att = nn.Sequential(
            QuaternionConv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            QuaternionBatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            QuaternionConv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            QuaternionBatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            QuaternionConv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            QuaternionBatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            QuaternionConv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            QuaternionBatchNorm1d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.local_att(x) + self.global_att(x)
        return self.sigmoid(att)


class MultiscaleQuaternionChannelAttentionModule(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        self.att = QuaternionChannelAttentionModule(channels, inter_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        alpha = self.att(x)
        return x * alpha


class QuaternionAttentionalFeatureFusion(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        self.att = QuaternionChannelAttentionModule(channels, inter_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        alpha = self.att(x1 + x2)
        return x1 * alpha + x2 * (1 - alpha)


class QuaternionIterativeAttentionalFeatureFusion(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.att1 = QuaternionChannelAttentionModule(in_channels, inter_channels)
        self.att2 = QuaternionChannelAttentionModule(in_channels, inter_channels)

    def forward(self, x1, x2):
        alpha = self.att1(x1 + x2)
        y_coarse = alpha * x1 + (1 - alpha) * x2
        beta = self.att2(y_coarse)
        y_refined = beta * x1 + (1 - beta) * x2
        return y_refined


QUATERNIONFUSIONBLOCKDICT = {name.lower(): obj for name, obj in globals().items() if inspect.isclass(obj) and issubclass(obj, nn.Module)}
