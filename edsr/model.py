import sys
sys.path.append("..")

import torch
from torch import nn

from srresnet.model import ResidualBlock


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class EDSR(nn.Module):
    """EDSR, modified from SRResNet"""

    def __init__(self, num_channels=3, num_scale=4, num_layers=32, within=False):
        super(EDSR, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)

        self.within = within
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

        self.conv1 = nn.Conv2d(num_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.res1 = self.MakeLayer(ResidualBlock, num_layers)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.upscale = nn.Sequential(*self.UpscaleBlock(256, 256 * 4, int(num_scale / 2)))

        self.conv3 = nn.Conv2d(256, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def MakeLayer(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block(channels=256, within=self.within))
        return nn.Sequential(*layers)

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
    def __init__(self, num_channels=3, num_scale=4, num_layers=32, within=False):
        super(VDSR, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)

        self.within = within
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

        self.conv1 = nn.Conv2d(num_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()

        self.res1 = self.MakeLayer(ResidualBlock, num_layers)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.upscale = nn.Sequential(*self.UpscaleBlock(256, 256 * 4, int(num_scale / 2)))

        self.conv3 = nn.Conv2d(256, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def MakeLayer(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block(channels=256, within=self.within))
        return nn.Sequential(*layers)

    def UpscaleBlock(self, in_channels, out_channels, num_scale):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale),
                  nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale), ]
        return layers

    def forward(self, x):
        out1 = x
        out = self.sub_mean(x)
        del x
        out = self.relu1(self.conv1(out))
        out = self.bn1(self.conv2(self.res1(out)))
        del out1
        out = self.conv3(self.upscale(out))
        out = self.add_mean(out)
        out = out + out1
        return out