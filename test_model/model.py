import sys

sys.path.append("..")

import torch
import torch.nn as nn
from srresnet.model import SRResNet


def pi(num=1000000000):
    sample = torch.rand(num, 2)
    dist = sample.norm(2, 1)
    ratio = (dist < 1).float().mean()
    return ratio * 4


def dct(x, norm=None):
    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().view(-1, n)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.rfft(v, 1)

    k = - torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * pi() / (2 * n)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= torch.sqrt(n) * 2
        V[:, 1:] /= torch.sqrt(n / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


class PreNorm(nn.Module):
    def __init__(self, out_channels, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(out_channels)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = dct_2d(dct_2d(x))
        return x


class SRFNet(SRResNet):
    def __init__(self, num_channels=3, out_channels=64, num_scale=4, num_layers=16, within=True):
        super(SRFNet, self).__init__()
        self.bn1 = PreNorm(out_channels, FNetBlock())
