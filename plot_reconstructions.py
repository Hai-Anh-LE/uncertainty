"""
plot_reconstructions.py - Simple, clean visual grid comparison on Kodak #23 (kodim23.png)
showing reconstructed images for all methods at their best evaluated quality / bit rate.
- No zoom-in insets.
- No uncertainty map.
- Clean image, method name, and scores (bpp, PSNR, SSIM).
- LaTeX Computer Modern typography.
"""
import os
import pathlib
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from compression_methods import (
    DCTCompression, 
    WaveletCompression, 
    StandardCodec, 
    NeuralModel
)
from metrics import psnr, ssim

REPO_ROOT = pathlib.Path(__file__).resolve().parent
IMG_PATH = REPO_ROOT / "data/kodak/kodim23.png"
OUT_FIG = REPO_ROOT / "results/visual_comparison_kodim23.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_visual_grid(img_path: pathlib.Path, out_path: pathlib.Path):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_pil = Image.open(img_path).convert("RGB")
    x = transforms.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)

    # All 9 methods evaluated at their best quality / bit-rate setting + Ground Truth
    methods = [
        # Row 1: Ground Truth + Top Neural Codecs
        (r"$\mathrm{Original\ (Ground\ Truth)}$", None),
        (r"$\mathrm{Cheng\ 2020\ (Attention)}$", NeuralModel("cheng2020_anchor", quality=6, device=DEVICE)),
        (r"$\mathrm{Minnen\ 2018\ (Autoregressive)}$", NeuralModel("mbt2018_mean", quality=6, device=DEVICE)),
        (r"$\mathrm{Ball\acute{e}\ 2018\ (Hyperprior)}$", NeuralModel("bmshj2018_hyperprior", quality=6, device=DEVICE)),
        (r"$\mathrm{Ball\acute{e}\ 2018\ (Factorized)}$", NeuralModel("bmshj2018_factorized", quality=6, device=DEVICE)),
        
        # Row 2: Standard Codecs & Classical Transforms
        (r"$\mathrm{WebP}$", StandardCodec("WEBP", quality=80)),
        (r"$\mathrm{JPEG\ 2000}$", StandardCodec("JPEG2000", quality=25)),
        (r"$\mathrm{JPEG}$", StandardCodec("JPEG", quality=80)),
        (r"$\mathrm{DWT\ (Wavelet\ db4)}$", WaveletCompression(threshold=0.02)),
        (r"$\mathrm{DCT\ (Block\ 8\times 8)}$", DCTCompression(quality=80)),
    ]

    # Typography settings
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.2), dpi=300)
    axes = axes.flatten()

    orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    for idx, (name, model) in enumerate(methods):
        ax = axes[idx]
        if model is None:
            rec_np = orig_np
            subtitle = r"$\mathrm{Reference\ Image}$"
        else:
            with torch.no_grad():
                x_hat, bpp_val = model.compress_decompress(x)
            p_val = psnr(x, x_hat)
            s_val = ssim(x, x_hat)
            rec_np = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            subtitle = rf"${bpp_val:.3f}\ \mathrm{{bpp}}\ |\ {p_val:.2f}\ \mathrm{{dB}}\ |\ \mathrm{{SSIM}}:\ {s_val:.4f}$"

        ax.imshow(rec_np)
        ax.set_title(f"{name}\n{subtitle}", pad=6)
        ax.axis("off")

    plt.suptitle(
        r"$\mathrm{Visual\ Reconstruction\ on\ Kodak\ 23\ at\ Best\ Quality\ Setting}$", 
        fontsize=14, y=0.98
    )
    plt.tight_layout()
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved clean visual comparison grid: {out_path}")

if __name__ == "__main__":
    generate_visual_grid(IMG_PATH, OUT_FIG)
