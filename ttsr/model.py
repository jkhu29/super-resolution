import sys

sys.path.append("..")

from srresnet.model import ResidualBlock
from edsr.model import EDSR, MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SFE(nn.Module):
    def __init__(self, num_res_layers=8, out_channels=64):
        super(SFE, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res1 = self.MakeLayer(ResidualBlock, num_res_layers, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def MakeLayer(self, block, num_layers, out_channels):
        layers = []
        for _ in range(num_layers):
            layers.append(block(channels=out_channels, within=False))
        return nn.Sequential(*layers)

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


class SearchTrans(nn.Module):
    def __init__(self):
        super(SearchTrans, self).__init__()

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2)
        # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold = F.normalize(lrsr_lv3_unfold, dim=1)
        # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)
        # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)
        # [N, H*W]

        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2) * 2, lrsr_lv3.size(3) * 2), kernel_size=(6, 6),
                       padding=2, stride=2) / (3. * 3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2) * 4, lrsr_lv3.size(3) * 4), kernel_size=(12, 12),
                       padding=4, stride=4) / (3. * 3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1


# TODO: similar to MDSR
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
        self.MainNet = EDSR(out_channels=64, num_layers=16)
        self.LTE = LTE(requires_grad=True)
        self.LTE_copy = LTE(requires_grad=False)  # used in transferal perceptual loss
        self.SearchTransfer = SearchTrans()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if type(sr) != type(None):
            # used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1
