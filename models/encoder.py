import torch
import torch.nn as nn


# Utility: Conv block
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


class SecretPreprocessor(nn.Module):
    def __init__(self, in_c=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_c, base),
            ConvBlock(base, base),
            ResidualBlock(base),
            ConvBlock(base, base)
        )

    def forward(self, s):
        return self.net(s)


class HidingNet(nn.Module):
    """Encoder: combine cover + secret features -> stego image"""
    def __init__(self, cover_c=3, secret_c=3, base=64):
        super().__init__()
        self.secret_prep = SecretPreprocessor(in_c=secret_c, base=base // 2)
        in_channels = cover_c + (base // 2)
        self.hiding = nn.Sequential(
            ConvBlock(in_channels, base),
            ResidualBlock(base),
            ConvBlock(base, base),
            ResidualBlock(base),
            ConvBlock(base, base),
            nn.Conv2d(base, cover_c, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # assume inputs normalized to [-1, 1]
        )

    def forward(self, cover, secret):
        s_feat = self.secret_prep(secret)
        x = torch.cat([cover, s_feat], dim=1)
        stego = self.hiding(x)
        return stego
