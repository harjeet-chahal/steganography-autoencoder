import argparse
from utils.metrics import compute_psnr, compute_ssim
from utils.visualization import save_quartet_grid




def load_ckpt(path, device):
ckpt = torch.load(path, map_location=device)
hider = HidingNet(cover_c=3, secret_c=3, base=64).to(device)
revealer = RevealNet(out_c=3, base=64).to(device)
hider.load_state_dict(ckpt['hider']); revealer.load_state_dict(ckpt['revealer'])
hider.eval(); revealer.eval()
return hider, revealer, ckpt




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', type=str, required=True)
ap.add_argument('--data_root', type=str, default='./_data')
ap.add_argument('--image_size', type=int, default=64)
ap.add_argument('--out_dir', type=str, default='./results')
args = ap.parse_args()


os.makedirs(args.out_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hider, revealer, _ = load_ckpt(args.ckpt, device)


transform = T.Compose([
T.Resize((args.image_size, args.image_size)),
T.ToTensor(),
T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
ds = CIFAR10(root=args.data_root, train=False, download=True, transform=transform)


from torch.utils.data import Dataset
import random


class EvalPairs(Dataset):
def __init__(self, base_ds):
self.ds = base_ds
def __len__(self):
return 200 # evaluate on 200 random pairs
def __getitem__(self, idx):
i = random.randint(0, len(self.ds) - 1)
j = random.randint(0, len(self.ds) - 1)
c, _ = self.ds[i]; s, _ = self.ds[j]
return c, s


loader = DataLoader(EvalPairs(ds), batch_size=8, shuffle=False)


total_psnr_cover, total_ssim_cover = 0.0, 0.0
total_psnr_secret, total_ssim_secret = 0.0, 0.0


with torch.no_grad():
first_saved = False
for cover, secret in loader:
cover = cover.to(device)
secret = secret.to(device)
stego = hider(cover, secret)
decoded = revealer(stego)


for k in range(cover.size(0)):
total_psnr_cover += compute_psnr(cover[k], stego[k])
total_ssim_cover += compute_ssim(cover[k], stego[k])
total_psnr_secret += compute_psnr(secret[k], decoded[k])
total_ssim_secret += compute_ssim(secret[k], decoded[k])


if not first_saved:
n = min(4, cover.shape[0])
save_quartet_grid(cover[:n], stego[:n], secret[:n], decoded[:n], os.path.join(args.out_dir, 'eval_quartet_grid.png'), nrow=n)
first_saved = True


N = len(loader.dataset)
print(f"Cover PSNR: {total_psnr_cover / N:.2f} dB | SSIM: {total_ssim_cover / N:.4f}")
print(f"Secret PSNR: {total_psnr_secret / N:.2f} dB | SSIM: {total_ssim_secret / N:.4f}")


if __name__ == '__main__':
main()