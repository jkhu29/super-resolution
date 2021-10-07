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

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
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
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, refer_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, refer_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, refer_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2) * 2, lrsr_lv3.size(3) * 2), kernel_size=(6, 6),
                       padding=2, stride=2) / (3. * 3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2) * 4, lrsr_lv3.size(3) * 4), kernel_size=(12, 12),
                       padding=4, stride=4) / (3. * 3.)

        s = refer_lv3_star.view(refer_lv3_star.size()[0], 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return s, T_lv3, T_lv2, T_lv1


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


class CSFI2(nn.Module):
    def __init__(self, out_channels):
        super(CSFI2, self).__init__()
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv21 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_merge1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_merge2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, out_channels):
        super(CSFI3, self).__init__()
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv21 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv31_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv31_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv32 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.conv_merge1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_merge2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_merge3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, out_channels):
        super(MergeTail, self).__init__()
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_merge = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_tail1 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_tail2 = nn.Conv2d(out_channels // 2, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)

        return x


class MainNet(nn.Module):
    def __init__(self, num_resblock, out_channels, res_scale):
        super(MainNet, self).__init__()
        self.num_resblocks = num_resblock
        self.out_channels = out_channels

        self.SFE = SFE(self.num_resblocks[0], out_channels)

        # stage11
        self.conv11_head = nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.RB11 = nn.ModuleList()
        for _ in range(self.num_resblocks[0]):
            self.RB11.append(ResidualBlock(out_channels))
        self.conv11_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # subpixel 1 -> 2
        self.conv12 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.ps12 = nn.PixelShuffle(2)

        # stage21, 22
        self.conv21_head = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv22_head = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.ex12 = CSFI2(out_channels)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for _ in range(self.num_resblocks[1]):
            self.RB21.append(ResidualBlock(out_channels))
            self.RB22.append(ResidualBlock(out_channels))

        self.conv21_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv22_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # subpixel 2 -> 3
        self.conv23 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.ps23 = nn.PixelShuffle(2)

        # stage31, 32, 33
        self.conv31_head = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv32_head = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv33_head = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.ex123 = CSFI3(out_channels)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for _ in range(self.num_resblocks[2]):
            self.RB31.append(ResidualBlock(out_channels))
            self.RB32.append(ResidualBlock(out_channels))
            self.RB33.append(ResidualBlock(out_channels))

        self.conv31_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv32_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv33_tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.merge_tail = MergeTail(out_channels)

    def forward(self, x, s, T_lv3, T_lv2, T_lv1):
        # shallow feature extraction
        x = self.SFE(x)

        # stage11
        x11 = x

        # soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res)  # F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * s
        x11 = x11 + x11_res

        x11_res = x11

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        # stage21, 22
        x21 = x11
        x21_res = x21
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))

        # soft-attention
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res)  # F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(s, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        # stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))

        # soft-attention
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res)  # F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(s, scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res

        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res

        x = self.merge_tail(x31, x32, x33)
        return x


class TTSR(nn.Module):
    def __init__(self):
        super(TTSR, self).__init__()
        self.num_res_blocks = [8, 12, 12]
        self.net = MainNet(num_resblock=self.num_res_blocks, out_channels=64, res_scale=0.1)
        self.lte = LTE(requires_grad=True)
        self.lte_copy = LTE(requires_grad=False)  # used in transferal perceptual loss
        self.trans = Trans()

    def forward(self, lr, lrsr, ref, refsr, sr=None):
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
