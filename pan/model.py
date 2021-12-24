import torch
import torch.nn as nn
import torch.nn.functional as F

from plugin import make_layer
from plugin.attention_modules import PixelAttention


class PixelAttentionConv(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super(PixelAttentionConv, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        # self.activate = nn.Sigmoid()
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.activate(res)
        x = torch.mul(self.conv2(x), res)
        x = self.conv3(x)
        return x


class SCPA(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 2):
        super(SCPA, self).__init__()
        reduc_channels = channels // reduction
        self.conv1_a = nn.Conv2d(channels, reduc_channels, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(channels, reduc_channels, kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(reduc_channels, reduc_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.pa = PixelAttentionConv(reduc_channels)

        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1_a(x))
        x2 = self.relu(self.conv1_b(x))

        x1 = self.relu(self.conv2(x1))
        x2 = self.relu(self.pa(x2))

        out = self.conv(torch.cat([x1, x2], dim=1)) + x

        return out


class PAN(nn.Module):
    def __init__(self, channels: int = 64, in_channels: int = 3, out_channels: int = 3, scale: int = 4, num_scpa: int = 2):
        super(PAN, self).__init__()
        self.scale = scale
        up_channels = channels * self.scale

        self.conv0 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.scpa_trunk = make_layer(SCPA, num_layers=num_scpa)
        self.trunk_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = nn.Conv2d(channels, up_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.pa = PixelAttention(up_channels)
        self.conv2 = nn.Conv2d(up_channels, up_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = nn.Conv2d(up_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x

        x = self.conv0(x)
        trunk = self.trunk_conv(self.scpa_trunk(x))
        x += trunk

        x = self.conv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
        x = self.activate(self.pa(x))
        x = self.activate(self.conv2(x))

        x = self.conv3(x)

        x_hr = F.interpolate(res, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x_hr + x
