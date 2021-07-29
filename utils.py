import torch.nn as nn
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def weights_init(model):
    """Official init from torch repo."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0.1)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.)


def calc_ssim(img1, img2):
    """calculate SSIM"""
    # TODO: convert to cuda
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return ssim(img1, img2, multichannel=True)
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image channel.')
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_pnsr(img1, img2):
    """calculate PNSR"""
    # TODO: convert to cuda
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1 = img1.numpy().transpose(1, 2, 0)
    img2 = img2.numpy().transpose(1, 2, 0)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return cv2.PSNR(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return cv2.PSNR(img1 * 255., img2 * 255.)
        elif img1.shape[2] == 1:
            return cv2.PSNR(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image channel.')
    else:
        raise ValueError('Wrong input image dimensions.')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
