import torch
import torch.nn as nn


class StegoLoss(nn.Module):
    """
    Weighted sum of cover-preservation and secret-reconstruction losses.
    Inputs/targets are assumed in [-1, 1].
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, cover, stego, secret, decoded):
        l_cover = self.mse(stego, cover)
        l_secret = self.mse(decoded, secret)
        total_loss = self.alpha * l_cover + self.beta * l_secret
        return total_loss, l_cover, l_secret
