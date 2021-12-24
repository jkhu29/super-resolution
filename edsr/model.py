import torch
import torch.nn as nn

from plugin import make_layer
from plugin.mean_shift import MeanShift
from plugin.commom import ResidualBlock


class EDSR(nn.Module):
    """EDSR, modified from SRResNet"""

    def __init__(self, num_channels=3, out_channels=64, num_scale=4, num_layers=8, within=False):
        super(EDSR, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)

        self.within = within
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

        self.conv1 = nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.res1 = make_layer(ResidualBlock, num_layers)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.upscale = nn.Sequential(*self.UpscaleBlock(out_channels, out_channels * 4, int(num_scale / 2)))

        self.conv3 = nn.Conv2d(out_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def UpscaleBlock(self, in_channels, out_channels, num_scale):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale),
                  nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale), ]
        return layers

    def forward(self, x):
        out = self.sub_mean(x)
        del x
        out = self.conv1(out)
        out1 = out
        out = self.bn1(self.conv2(self.res1(out1)))
        out = out + out1
        del out1
        out = self.conv3(self.upscale(out))
        out = self.add_mean(out)
        return out


class VDSR(nn.Module):
    """VDSR, modified from EDSR"""
    def __init__(self, num_channels=3, num_scale=4, num_layers=8, within=False):
        super(VDSR, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)

        self.within = within
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()

        self.res1 = make_layer(ResidualBlock, num_layers)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.upscale = nn.Sequential(*self.UpscaleBlock(64, 64 * 4, int(num_scale // 2)))

        self.conv3 = nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def UpscaleBlock(self, in_channels, out_channels, num_scale):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale),
                  nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale), ]
        return layers

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.relu1(self.conv1(x))
        res = x
        x = self.bn1(self.conv2(self.res1(x))) + res
        x = self.conv3(self.upscale(x))
        x = self.add_mean(x)
        return x
