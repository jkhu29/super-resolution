import torch
import torch.nn as nn


def _make_layer(block, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels: int = 64, out_channels: int = 64, withbn=True):
        super(ConvReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvPixelShuffle(nn.Module):
    """ConvPixelShuffle"""
    def __init__(self, in_channels: int = 64, out_channels: int = 64, num_scale=2):
        super(ConvPixelShuffle, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_scale**2 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(num_scale)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x


class ABBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size1: int = 3,
        padding1: int = 0,
        kernel_size2: int = 1,
        padding2: int = 1,
    ):
        super(ABBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size2, kernel_size1), padding=(padding1, padding2), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size1, kernel_size2), padding=(padding2, padding1), bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size1, kernel_size1), padding=(padding2, padding2), bias=False)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return self.relu(x)


class AcNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        kernel_size1: int = 3,
        padding1: int = 0,
        kernel_size2: int = 1,
        padding2: int = 1,
        scale: int = 4,
        multi_scale: int = 4,
    ):
        super(AcNet, self).__init__()
        self.conv = ABBlock(in_channels=3)
        self.ab1 = _make_layer(
            ABBlock, num_layers=16,
        )
        self.relu = nn.CELU(inplace=True)
        self.upsample = ConvPixelShuffle()

        self.conv11 = ConvReLU()
        self.conv12 = ConvReLU()
        self.conv21 = ConvReLU()
        self.conv22 = ConvReLU()

        self.conv30 = ConvReLU()
        self.conv31 = ConvReLU()
        self.conv32 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        res = x
        x = self.ab1(x)
        x += res

        res = self.upsample(res)
        x = self.upsample(x)

        res = self.conv11(res)
        x = self.conv12(x)
        res = self.conv21(res)
        x = self.conv22(x)
        x += res
        x = self.relu(x)
        x = self.conv30(x)
        x = self.conv31(x)
        x = self.conv32(x)
        return x


if __name__ == "__main__":
    a = AcNet()
    test_data = torch.rand(2, 3, 128, 128)
    a(test_data)
