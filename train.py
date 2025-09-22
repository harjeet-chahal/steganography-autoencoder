import argparse




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--data_root', type=str, default='./_data')
ap.add_argument('--epochs', type=int, default=50)
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--image_size', type=int, default=64)
ap.add_argument('--alpha', type=float, default=0.7)
ap.add_argument('--beta', type=float, default=0.3)
ap.add_argument('--noise_std', type=float, default=0.0, help='Gaussian noise between encoder and decoder')
ap.add_argument('--out_dir', type=str, default='./results')
ap.add_argument('--seed', type=int, default=1337)
args = ap.parse_args()


os.makedirs(args.out_dir, exist_ok=True)


# Repro
random.seed(args.seed)
torch.manual_seed(args.seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data
train_ds = PairCIFAR(args.data_root, train=True, image_size=args.image_size)
val_ds = PairCIFAR(args.data_root, train=False, image_size=args.image_size)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)


# Models
hider = HidingNet(cover_c=3, secret_c=3, base=64).to(device)
revealer = RevealNet(out_c=3, base=64).to(device)


# Loss + Optim
criterion = StegoLoss(alpha=args.alpha, beta=args.beta)
opt = optim.Adam(list(hider.parameters()) + list(revealer.parameters()), lr=args.lr)


best_val = float('inf')
for epoch in range(1, args.epochs + 1):
tr, tr_c, tr_s = train_one_epoch(hider, revealer, train_loader, opt, device, criterion, args.noise_std)
va, va_c, va_s = evaluate(hider, revealer, val_loader, device, criterion, save_dir=args.out_dir)


print(f"[Epoch {epoch:03d}] train: {tr:.4f} (cover {tr_c:.4f} | secret {tr_s:.4f}) | val: {va:.4f} (cover {va_c:.4f} | secret {va_s:.4f})")


# Checkpointing
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