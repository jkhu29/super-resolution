import torch
import torch.nn.functional as F


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = F.pairwise_distance(x, y, p=1)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class L2_Charbonnier_loss(torch.nn.Module):
    """L2 Charbonnierloss."""
    def __init__(self):
        super(L2_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = F.pairwise_distance(x, y, p=2)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss
