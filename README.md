# Deep Learning Autoencoder for Image Steganography


Hide a secret image inside a cover image (visually indistinguishable) and recover it, using a convolutional autoencoder.


## ✨ Features
- Hiding (encoder) + revealing (decoder) networks.
- Weighted loss: preserve cover fidelity + recover secret.
- Metrics: PSNR, SSIM (cover↔stego, secret↔decoded).
- Robustness option: Gaussian noise between encoder/decoder.
- Streamlit demo for interactive uploads.


## 🏗️ Architecture
- **HidingNet**: secret preprocessor → concat with cover → conv/residual stack → stego.
- **RevealNet**: conv/residual stack → decoded secret.
- Loss: `L = α·MSE(stego, cover) + β·MSE(decoded, secret)` (default α=0.7, β=0.3).


## 🚀 Quickstart
```bash
# 1) Install deps
pip install -r requirements.txt


# 2) Train (CIFAR-10 pairs, 64×64)
python train.py --epochs 50 --batch_size 64 --lr 1e-4 --image_size 64 --out_dir results


# 3) Evaluate
python evaluate.py --ckpt results/best.pt --out_dir results


# 4) Demo (after training)
streamlit run demo.py

## 🔜 Upcoming
- Switch dataset to CelebA for higher resolution images
- Add adversarial robustness module
