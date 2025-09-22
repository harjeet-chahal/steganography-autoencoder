import os
import torch
import torchvision.utils as vutils


def save_quartet_grid(cover, stego, secret, decoded, out_path, nrow=4):
    """
    Save a grid of cover, stego, secret, decoded images for comparison.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def clamp(x):
        return torch.clamp(x, -1, 1)

    grid = vutils.make_grid(
        torch.cat([clamp(cover), clamp(stego), clamp(secret), clamp(decoded)], dim=0),
        nrow=nrow,
        normalize=True,
        range=(-1, 1)
    )
    vutils.save_image(grid, out_path)
