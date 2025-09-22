import io
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as T

from models import HidingNet, RevealNet


@st.cache_resource
def load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hider = HidingNet(3, 3, base=64).to(device)
    revealer = RevealNet(3, base=64).to(device)
    hider.load_state_dict(ckpt['hider'])
    revealer.load_state_dict(ckpt['revealer'])
    hider.eval()
    revealer.eval()
    return hider, revealer


def tform(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def to_pil(x):
    """
    Convert tensor [-1,1] to PIL Image
    """
    x = (x.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    x = (x * 255).byte()
    return T.ToPILImage()(x)


def main():
    st.set_page_config(page_title='Steganography Autoencoder Demo', layout='centered')
    st.title('🔒 Steganography Autoencoder')
    st.caption('Hide a secret image inside a cover image, then reveal it.')

    ckpt_path = st.text_input('Path to checkpoint (.pt)', './results/best.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    col1, col2 = st.columns(2)
    with col1:
        cover_file = st.file_uploader('Upload Cover Image', type=['png', 'jpg', 'jpeg'])
    with col2:
        secret_file = st.file_uploader('Upload Secret Image', type=['png', 'jpg', 'jpeg'])

    size = st.slider('Model image size', min_value=64, max_value=256, value=64, step=32)

    if st.button('Run'):
        if not ckpt_path:
            st.error('Please provide a checkpoint path.')
            return
        try:
            hider, revealer = load_models(ckpt_path, device)
        except Exception as e:
            st.error(f'Failed to load checkpoint: {e}')
            return

        tf = tform(size)
        if cover_file is None or secret_file is None:
            st.warning('Please upload both cover and secret images.')
            return

        cover_img = Image.open(io.BytesIO(cover_file.read())).convert('RGB')
        secret_img = Image.open(io.BytesIO(secret_file.read())).convert('RGB')

        cover_t = tf(cover_img).unsqueeze(0).to(device)
        secret_t = tf(secret_img).unsqueeze(0).to(device)

        with torch.no_grad():
            stego = hider(cover_t, secret_t)
            decoded = revealer(stego)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.subheader('Cover vs. Stego')
            st.image([cover_img.resize((size, size)), to_pil(stego[0])],
                     caption=['Cover', 'Stego'])
        with sc2:
            st.subheader('Secret vs. Decoded')
            st.image([secret_img.resize((size, size)), to_pil(decoded[0])],
                     caption=['Secret', 'Decoded'])


if __name__ == '__main__':
    main()
