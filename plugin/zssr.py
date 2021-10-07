import torch.nn as nn


class ZSSR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=5):
        super(ZSSR, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            ]
        layers += [nn.Conv2d(out_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x) + x
        return x
