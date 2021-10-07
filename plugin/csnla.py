import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.relu1 = nn.CELU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1)
        self.relu2 = nn.CELU(inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.CELU(inplace=True)

    def forward(self, x):
        x_embed1 = self.relu1(self.conv1(x))
        x_embed2 = self.relu2(self.conv2(x))
        x_assembly = self.relu3(self.conv3(x))

        n, c, h, w = x_embed1.shape
        x_embed1 = x_embed1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed2 = x_embed2.view(n, c, h * w)
        score = torch.matmul(x_embed1, x_embed2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(n, -1, h * w).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly).permute(0, 2, 1).view(n, -1, h, w)

        return score, x_final