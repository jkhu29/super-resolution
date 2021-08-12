import torch
import torch.nn as nn

import sys
sys.path.append("..")

from edsr.model import MeanShift
from srresnet.model import ResidualBlock
from rcan.model import ChannelAttention, RCAB, RG


class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class FFTBlock(nn.Module):
    def __init__(self, channel=64):
        super(FFTBlock, self).__init__()
        self.conv_fc = nn.Sequential(
            nn.Conv2d(1, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1, padding=0, bias=True),
            nn.Softplus()
        )

    def forward(self, x, u, d, sigma):
        rho = self.conv_fc(sigma)
        x = torch.irfft(
            self.divcomplex(
                        u + rho.unsqueeze(-1)*torch.rfft(x, 2, onesided=False
                    ),
                d + self.real2complex(rho)), 2, onesided=False
            )
        return x

    def divcomplex(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = y[..., 1]
        cd2 = c**2 + d**2
        return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)

    def real2complex(self, x):
        return torch.stack([x, torch.zeros(x.shape).type_as(x)], -1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        super(ResidualDenseBlock, self).__init__()
        assert in_channels % 2 == 0
        half_in_channels = in_channels // 2

        # gc: growth channel
        self.conv1 = nn.Conv2d(in_channels, half_in_channels, kernel_size, stride, padding, bias)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(in_channels+half_in_channels, half_in_channels, kernel_size, stride, padding, bias)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(in_channels+2*half_in_channels, half_in_channels, kernel_size, stride, padding, bias)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(in_channels+3*half_in_channels, half_in_channels, kernel_size, stride, padding, bias)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(in_channels+4*half_in_channels, in_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = self.relu2(x2)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x3 = self.relu2(x3)
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = self.relu2(x4)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


class RRDB(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, stride=1, padding=1, bias=False):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock(in_channels, kernel_size, stride, padding, bias)
        self.RDB2 = ResidualDenseBlock(in_channels, kernel_size, stride, padding, bias)
        self.RDB3 = ResidualDenseBlock(in_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x
