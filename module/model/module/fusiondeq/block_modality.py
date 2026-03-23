import torch
from torch import nn
import inspect

class SmallBlock(nn.Module):
    def __init__(self, out_dim, kernel_size, bias):
        """
        Pre-block for DEQEQFusionModule.
        """
        super(SmallBlock, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.conv2 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.gn2 = torch.nn.BatchNorm1d(self.out_dim)  # nn.GroupNorm(1, inplanes, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, injection_feature):
        out = self.conv1(x) + injection_feature
        out = self.relu2(self.conv2(self.gn2(out)))
        return out


class SmallGatedBlock(nn.Module):
    def __init__(self, out_dim, kernel_size, bias):
        """
        Pre-block for DEQEQFusionModule.
        """
        super(SmallGatedBlock, self).__init__()
        self.out_dim = out_dim
        self.conv1 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.conv2 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.gn2 = torch.nn.BatchNorm1d(self.out_dim)  # nn.GroupNorm(1, inplanes, affine=True)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Tanh()

    def forward(self, x, injection_feature):
        out = self.activation1(self.conv1(x)) * injection_feature
        out = self.activation2(self.conv2(self.gn2(out)))
        return out


class MiddleGatedAddBlock(nn.Module):
    def __init__(self, out_dim, kernel_size, bias):
        """
        Pre-block for DEQEQFusionModule.
        """
        super(MiddleGatedAddBlock, self).__init__()
        self.out_dim = out_dim
        self.conv1 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.activation1 = nn.Sigmoid()
        self.conv2 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.gn2 = torch.nn.BatchNorm1d(self.out_dim)
        self.conv3 = torch.nn.Conv1d(
            self.out_dim, self.out_dim, kernel_size=kernel_size, padding="same", bias=bias
        )
        self.gn3 = torch.nn.BatchNorm1d(self.out_dim)
        self.activation3 = nn.Tanh()

    def forward(self, x, injection_feature):
        out = self.activation1(self.conv1(x)) * injection_feature
        out = self.conv2(self.gn2(out)) + injection_feature
        out = self.activation3(self.conv3(self.gn3(out)))
        # print(x.sum(), out.sum())
        return out

MODALITYBLOCKDICT = {
    name.lower(): obj
    for name, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, nn.Module)
}
