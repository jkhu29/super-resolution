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


# thanks: https://github.com/dfdazac/wassdistance/blob/master/layers.py
def sinkhorn_img_normalized(x, y, device, epsilon=0.01, niter=100):
    Wxy = sinkhorn_img_loss(x, y, device, epsilon=epsilon, niter=niter)
    Wxx = sinkhorn_img_loss(x, x, device, epsilon=epsilon, niter=niter)
    Wyy = sinkhorn_img_loss(y, y, device, epsilon=epsilon, niter=niter)
    return 2 * Wxy - Wxx - Wyy


def cost_matrix(x, y, p=2):
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return c


def M(C, u, v, epsilon):
    return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon


def sinkhorn_img_loss(X, Y, device, epsilon=0.01, niter=100):
    """
    X: [N C H W]
    Y: [N C H W]
    X.shaye == Y.shape
    """
    assert X.shape == Y.shape, "shape of x and y should be the same, " \
                               "but get x: {}, y: {}".format(X.shape, Y.shape)

    channels = X.shape[1]
    cost = torch.zeros(X.shape[0], channels).to(device)

    for i in range(channels):
        x = X[:, i, :, :].squeeze()
        y = Y[:, i, :, :].squeeze()

        C = cost_matrix(x, y)
        batch_size = x.shape[0]

        mu = torch.ones(batch_size, x.shape[-2], requires_grad=False).squeeze().to(device) / x.shape[-2]
        nu = torch.ones(batch_size, x.shape[-2], requires_grad=False).squeeze().to(device) / x.shape[-2]

        u, v, err = 0. * mu, 0. * nu, 0.
        actual_nits = 0
        thresh = 0.1
        # to check if algorithm terminates because of threshold or max iterations reached

        for _ in range(niter):
            u1 = u
            # useful to check the update
            u += epsilon * (torch.log(mu + 1e-8) - torch.logsumexp(M(C, u, v, epsilon), dim=-1))
            v += epsilon * (torch.log(nu + 1e-8) - torch.logsumexp(M(C, u, v, epsilon).transpose(-2, -1), dim=-1))
            # accelerated unbalanced iterations
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        pi = torch.exp(M(C, u, v, epsilon))
        # Transport plan pi = diag(a)*K*diag(b)
        cost[:, i] = torch.sum(pi * C, dim=(-2, -1))
        # Sinkhorn cost

    return cost


class L1_W_loss(torch.nn.Module):
    """L1 W loss."""

    def __init__(self):
        super(L1_W_loss, self).__init__()

    def forward(self, x, y):
        loss = sinkhorn_img_loss(x, y, x.device)
        return loss.sum()


class L2_W_loss(torch.nn.Module):
    """L2 W loss."""

    def __init__(self):
        super(L2_W_loss, self).__init__()

    def forward(self, x, y):
        loss = sinkhorn_img_normalized(x, y, x.device)
        return loss.sum()


class L1_Charbonnier_W_loss(torch.nn.Module):
    """L1 Charbonnier W loss."""

    def __init__(self):
        super(L1_Charbonnier_W_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = sinkhorn_img_loss(x, y, x.device)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class L2_Charbonnier_W_loss(torch.nn.Module):
    """L2 Charbonnier W loss."""

    def __init__(self):
        super(L2_Charbonnier_W_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = sinkhorn_img_normalized(x, y, x.device)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class L1_edge_W_loss(torch.nn.Module):
    """TODO: L1 W loss."""

    def __init__(self):
        super(L1_edge_W_loss, self).__init__()
        kernel = [[[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]],
                  [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]],
                  [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]]
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0)
        self.weight = torch.nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x, y, threshold=127):
        x = F.conv2d(x, self.weight, padding=2)
        y = F.conv2d(y, self.weight, padding=2)
        # x = torch.where(x < threshold, 0, 255)
        # y = torch.where(y < threshold, 0, 255)

        loss = sinkhorn_img_loss(x, y, self.device)
        return loss.sum()
