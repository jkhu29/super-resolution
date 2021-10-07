import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=True, kernel_size=3, stride=1, padding=1):
        super(ConvReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """ResBlock used by CartoonGAN and DeCartoonGAN"""
    def __init__(self, num_conv=1, channels=64):
        super(ResBlock, self).__init__()

        self.conv_relu = ConvReLU(channels, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        x = self.conv(x) + res
        return x


class ESA(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 4):
        super(ESA, self).__init__()
        mid_channels = channels // reduction
        self.conv1 = ConvReLU(channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_f = ConvReLU(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_max = ConvReLU(mid_channels, mid_channels)
        self.conv2 = ConvReLU(mid_channels, mid_channels, kernel_size=3, stride=2, padding=0)
        self.conv3 = ResBlock(mid_channels, mid_channels)
        self.conv4 = ConvReLU(mid_channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)
        x = self.conv_max(x)
        x = self.conv3(x)
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
        res = self.conv_f(res)
        x = self.conv4(x + res)
        m = torch.sigmoid(x)
        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels: int = 64, reduction: int = 4):
        super(RFDB, self).__init__()
        distilled_channels = in_channels // reduction
        remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels, distilled_channels, kernel_size=1, padding=0)
        self.c1_r = nn.Conv2d(in_channels, remaining_channels, kernel_size=3, stride=1, padding=1)
        self.c2_d = nn.Conv2d(remaining_channels, distilled_channels, kernel_size=1, padding=0)
        self.c2_r = nn.Conv2d(remaining_channels, remaining_channels, kernel_size=3, stride=1, padding=1)
        self.c3_d = nn.Conv2d(remaining_channels, distilled_channels, kernel_size=1, padding=0)
        self.c3_r = nn.Conv2d(remaining_channels, remaining_channels, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(remaining_channels, distilled_channels, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(distilled_channels * 4, in_channels, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.esa = ESA(in_channels)

    def forward(self, x):
        distilled_c1 = self.c1_d(x)
        r_c1 = self.c1_r(x)
        r_c1 = self.act(r_c1+x)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused


class RFDN(nn.Module):
    def __init__(self, channels: int = 64, scale: int = 4):
        super(RFDN, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, 1, 1)
        self.rfdb1 = RFDB(channels)
        self.rfdb2 = RFDB(channels)
        self.rfdb3 = RFDB(channels)
        self.rfdb4 = RFDB(channels)

        self.conv2 = ConvReLU(channels * 4, channels, kernel_size=1, padding=0)
        self.lr_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.upsample = nn.Sequential(
            ConvReLU(channels, 3 * (scale ** 2)),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x1 = self.rfdb1(x)
        x2 = self.rfdb2(x1)
        x3 = self.rfdb2(x2)
        x4 = self.rfdb2(x3)
        x = self.conv2(torch.cat([x1, x2, x3, x4], dim=1))
        x = self.lr_conv(x) + res
        x = self.upsample(x)
        return x
