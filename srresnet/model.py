import torch
from torch import nn

from plugin import make_layer
from plugin.commom import ResidualBlock


class SRCNN(nn.Module):
    """SRCNN"""
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


class FSRCNN(nn.Module):
    """FSRCNN"""
    def __init__(self, num_channels=3, m=4):
        super(FSRCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 56, kernel_size=5, padding=5 // 2),
            nn.PReLU(56),
            )
        self.conv2 = [nn.Conv2d(56, 12, kernel_size=1), nn.PReLU(12)]
        for _ in range(m):
            self.conv2.extend([nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2), nn.PReLU(12)])
        self.conv2.extend([nn.Conv2d(12, 56, kernel_size=1), nn.PReLU(56)])
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.ConvTranspose2d(56, num_channels, kernel_size=9, padding=9 // 2)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class BSRCNN(nn.Module):
    """SRCNN with BatchNorm/InstanceNorm"""
    def __init__(self, num_channels=3, m=4):
        super(BSRCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 56, kernel_size=5, padding=5 // 2),
            nn.PReLU(56)
            )
        self.conv2 = [nn.Conv2d(56, 12, kernel_size=1), nn.PReLU(12)]
        for _ in range(m):
            self.conv2.extend([nn.Conv2d(12, 12, kernel_size=1), nn.InstanceNorm2d(12, affine=True), nn.PReLU(16)])
        self.conv2.extend([nn.Conv2d(12, 56, kernel_size=1), nn.PReLU(56)])
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.ConvTranspose2d(56, num_channels, kernel_size=9, padding=9 // 2)
        
    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class SRResNet(nn.Module):
    """SRResNet with or without BN"""
    def __init__(self, num_channels=3, out_channels=64, num_scale=4, num_layers=16, within=True):
        super(SRResNet, self).__init__()

        self.within = within

        self.conv1 = nn.Conv2d(num_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.LeakyReLU(0.2)

        self.res1 = make_layer(ResidualBlock, num_layers)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.upscale = nn.Sequential(*self.UpscaleBlock(out_channels, out_channels * 2 ** 2, num_scale))

        self.conv3 = nn.Conv2d(16, num_channels, kernel_size=9, stride=1, padding=4)

    def UpscaleBlock(self, in_channels, out_channels, num_scale):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.PixelShuffle(num_scale),
                  nn.LeakyReLU(0.2)]
        return layers

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        del x
        out = self.bn1(self.conv2(self.res1(out1)))
        out = out + out1
        del out1
        out = self.conv3(self.upscale(out))
        return out
