import sys

sys.path.append("..")

from srresnet.model import ResidualBlock
from edsr.model import EDSR, MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _make_layer(block, num_layers, out_channels):
    layers = []
    for _ in range(num_layers):
        layers.append(block(channels=out_channels, within=False))
    return nn.Sequential(*layers)


class SFE(nn.Module):
    def __init__(self, num_res_layers=8, out_channels=64):
        super(SFE, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1 = _make_layer(ResidualBlock, num_res_layers, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        del x
        out1 = out
        out = self.res1(out)
        out = self.conv2(out)
        out += out1
        return out


def bis(inputs, dim, index):
    views = [inputs.size(0)] + [1 if i != dim else -1 for i in range(1, len(inputs.size()))]
    expanse = list(inputs.size())
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inputs, dim, index)


class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv3):
        lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
        # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)
        # [N, C*k*k, H*W]

        refer_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)
        # [N, Hr*Wr, H*W]
        refer_lv3_star, refer_lv3_star_arg = torch.max(refer_lv3, dim=1)
        # [N, H*W]

        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)

        trans_lv3_unfold = bis(ref_lv3_unfold, 2, refer_lv3_star_arg)

        trans_lv3 = F.fold(trans_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1)

        s = refer_lv3_star.view(refer_lv3_star.size()[0], 1, lrsr_lv3.size()[2], lrsr_lv3.size()[3])

        return s, trans_lv3


class LTE(nn.Module):
    def __init__(self, requires_grad=True):
        super(LTE, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.add_mean = MeanShift(rgb_mean, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = self.add_mean(x)
        x = self.slice2(x)
        x_lv2 = self.add_mean(x)
        x = self.slice3(x)
        x_lv3 = self.add_mean(x)
        return x_lv1, x_lv2, x_lv3


class TTSR(nn.Module):
    def __init__(self):
        super(TTSR, self).__init__()
        self.num_res_blocks = [8, 12, 12]
        self.net = EDSR(out_channels=64, num_layers=16)
        self.lte = LTE(requires_grad=True)
        self.lte_copy = LTE(requires_grad=False)  # used in transferal perceptual loss
        self.trans = Trans()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if sr is not None:
            # used in transferal perceptual loss
            self.lte_copy.load_state_dict(self.lte.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.lte_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3 = self.lte((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.lte((refsr.detach() + 1.) / 2.)
        _, _, ref_lv3 = self.lte((ref.detach() + 1.) / 2.)

        s, trans_lv3 = self.trans(lrsr_lv3, refsr_lv3, ref_lv3)

        sr = self.net(lr, s, trans_lv3)

        return sr, s, trans_lv3
