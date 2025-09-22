import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBlock(c, c)
        self.conv2 = ConvBlock(c, c)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class RevealNet(nn.Module):
    """Decoder: stego -> reconstructed secret"""
    def __init__(self, out_c=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, base),
            ResidualBlock(base),
            ConvBlock(base, base),
            ResidualBlock(base),
            ConvBlock(base, base),
            nn.Conv2d(base, out_c, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, stego):
        return self.net(stego)
