import torch
from torch import nn
import inspect

class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = int(channels**0.5)
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        att = self.local_att(x) + self.global_att(x)
        return self.sigmoid(att)


class MultiscaleChannelAttentionModule(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        self.att = ChannelAttentionModule(channels, inter_channels)

    def forward(self, x):
        alpha = self.att(x)
        return x * alpha


class AttentionalFeatureFusion(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        self.att = ChannelAttentionModule(channels, inter_channels)

    def forward(self, x1, x2):
        alpha = self.att(x1 + x2)
        return x1 * alpha + x2 * (1 - alpha)


class IterativeAttentionalFeatureFusion(nn.Module):
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        self.att1 = ChannelAttentionModule(channels, inter_channels)
        self.att2 = ChannelAttentionModule(channels, inter_channels)

    def forward(self, x1, x2):
        alpha = self.att1(x1 + x2)
        y_coarse = alpha * x1 + (1 - alpha) * x2
        beta = self.att2(y_coarse)
        y_refined = beta * x1 + (1 - beta) * x2
        return y_refined


class DeepIterativeAttentionalFeatureFusion(nn.Module):
    def __init__(self, channels, inter_channels=None, repeat_times=3):
        super().__init__()
        self.att = nn.ModuleList([ChannelAttentionModule(channels, inter_channels) for _ in range(repeat_times)])

    def forward(self, x1, x2):
        x = x1 + x2
        for att in self.att:
            alpha = att(x)
            x = alpha * x1 + (1 - alpha) * x2
        return x


BASEFUSIONBLOCKDICT = {name.lower(): obj for name, obj in globals().items() if inspect.isclass(obj) and issubclass(obj, nn.Module)}
