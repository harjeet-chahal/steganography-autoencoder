import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Inputs expected in [-1,1], CxHxW torch tensors


def to_img(x):
# x: torch tensor [-1,1]
x = x.detach().cpu().numpy()
return np.transpose((x + 1.0) / 2.0, (1, 2, 0)).clip(0, 1)




def compute_psnr(img1_t, img2_t):
i1, i2 = to_img(img1_t), to_img(img2_t)
return float(psnr(i1, i2, data_range=1.0))




def compute_ssim(img1_t, img2_t):
i1, i2 = to_img(img1_t), to_img(img2_t)
# multichannel=True is deprecated; channel_axis=-1 instead
return float(ssim(i1, i2, channel_axis=-1, data_range=1.0))