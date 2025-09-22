import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from models import HidingNet, RevealNet
from utils.losses import StegoLoss
from utils.visualization import save_quartet_grid


# --- Dataset wrapper: random (cover, secret) pairs ---
class PairCIFAR(Dataset):
    def __init__(self, root, train=True, image_size=64):
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # to [-1, 1]
        ])
        self.ds = CIFAR10(root=root, train=train, download=True, transform=transform)
        self.len = len(self.ds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cover, _ = self.ds[idx]
        j = random.randint(0, self.len - 1)
        secret, _ = self.ds[j]
        return cover, secret


def train_one_epoch(hider, revealer, loader, opt, device, criterion, noise_std=0.0):
    hider.train()
    revealer.train()
    total, c_accum, s_accum = 0.0, 0.0, 0.0

    for cover, secret in tqdm(loader, desc="Training", leave=False):
        cover = cover.to(device)
        secret = secret.to(device)

        opt.zero_grad(set_to_none=True)
        stego = hider(cover, secret)

        if noise_std > 0:
            stego_noisy = stego + noise_std * torch.randn_like(stego)
            decoded = revealer(stego_noisy)
        else:
            decoded = revealer(stego)

        loss, l_cover, l_secret = criterion(cover, stego, secret, decoded)
        loss.backward()
        nn.utils.clip_grad_norm_(list(hider.parameters()) + list(revealer.parameters()), 1.0)
        opt.step()

        total += float(loss.item())
        c_accum += float(l_cover.item())
        s_accum += float(l_secret.item())

    n = len(loader)
    return total / n, c_accum / n, s_accum / n


def evaluate(hider, revealer, loader, device, criterion, save_dir=None):
    hider.eval()
    revealer.eval()
    total, c_accum, s_accum = 0.0, 0.0, 0.0
    example_saved = False

    with torch.no_grad():
        for cover, secret in tqdm(loader, desc="Evaluating", leave=False):
            cover = cover.to(device)
            secret = secret.to(device)
            stego = hider(cover, secret)
            decoded = revealer(stego)

            loss, l_cover, l_secret = criterion(cover, stego, secret, decoded)
            total += float(loss.item())
            c_accum += float(l_cover.item())
            s_accum += float(l_secret.item())

            if (save_dir is not None) and (not example_saved):
                k = min(4, cover.shape[0])
                save_quartet_grid(
                    cover[:k], stego[:k], secret[:k], decoded[:k],
                    os.path.join(save_dir, 'val_quartet_grid.png'),
                    nrow=k
                )
                example_saved = True

    n = len(loader)
    return total / n, c_accum / n, s_accum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./_data')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--image_size', type=int, default=64)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--noise_std', type=float, default=0.0)
    ap.add_argument('--out_dir', type=str, default='./results')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = PairCIFAR(args.data_root, train=True, image_size=args.image_size)
    val_ds = PairCIFAR(args.data_root, train=False, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    hider = HidingNet(cover_c=3, secret_c=3, base=64).to(device)
    revealer = RevealNet(out_c=3, base=64).to(device)

    criterion = StegoLoss(alpha=args.alpha, beta=args.beta)
    opt = optim.Adam(list(hider.parameters()) + list(revealer.parameters()), lr=args.lr)

    best_val = float('inf')

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        tr, tr_c, tr_s = train_one_epoch(hider, revealer, train_loader, opt, device, criterion, args.noise_std)
        va, va_c, va_s = evaluate(hider, revealer, val_loader, device, criterion, save_dir=args.out_dir)

        print(f"[Epoch {epoch:03d}] "
              f"train: {tr:.4f} (cover {tr_c:.4f} | secret {tr_s:.4f}) | "
              f"val: {va:.4f} (cover {va_c:.4f} | secret {va_s:.4f})")

        ckpt = {
            'hider': hider.state_dict(),
            'revealer': revealer.state_dict(),
            'epoch': epoch,
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join(args.out_dir, 'last.pt'))
        if va < best_val:
            best_val = va
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))


if __name__ == '__main__':
    main()
