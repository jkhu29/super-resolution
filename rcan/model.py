import torch
import torch.nn as nn
from plugin.mean_shift import MeanShift
from plugin.attention_modules import ChannelAttention


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, scale=4, num_features=64, num_rg=5, num_rcab=10, reduction=8):
        super(RCAN, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.scale = int(scale / 2)

        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)  # rethink

        self.upscale = nn.Sequential(*self.UpscaleBlock(num_features, num_features * (self.scale ** 2), self.scale))
        self.conv3 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def UpscaleBlock(self, in_channels, out_channels, num_scale):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                  nn.PixelShuffle(num_scale),
                  nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                  nn.PixelShuffle(num_scale), ]
        return layers

    def forward(self, x, scale=0.1):
        out = self.conv1(self.sub_mean(x))
        del x
        out1 = self.bn1(self.conv2(self.rgs(out)))
        out = out + out1 * scale
        del out1
        out = self.conv3(self.upscale(out))
        out = self.add_mean(out)
        return out
