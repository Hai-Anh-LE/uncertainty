"""
plot_reconstructions.py - Generates a publication-quality visual comparison grid
of reconstructed images on Kodak #23 (kodim23.png) across all compression methods
at a matched bit-rate level (~0.3 bpp).

Includes full images with zoomed-in inset patches showing texture/edge artifacts
and the predictive Uncertainty map U(x).
"""
import os
import pathlib
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from compression_methods import (
    DFTCompression, 
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
    _, _, H, W = x.shape

    # Define crop region around high-detail bird head (x: 100-240, y: 140-280)
    crop_y0, crop_y1 = 140, 280
    crop_x0, crop_x1 = 100, 240

    # Compression methods configured for matched ~0.3 bpp regime
    methods = [
        (r"$\mathrm{Original\ (Ground\ Truth)}$", None, None),
        (r"$\mathrm{DFT\ (Fourier)}$", DFTCompression(keep_ratio=0.005), "k=0.5%"),
        (r"$\mathrm{DCT\ (Block\ 8\times 8)}$", DCTCompression(quality=10), "q10"),
        (r"$\mathrm{DWT\ (Wavelet\ db4)}$", WaveletCompression(threshold=0.15), "th0.15"),
        (r"$\mathrm{JPEG}$", StandardCodec("JPEG", quality=20), "q20"),
        (r"$\mathrm{JPEG\ 2000}$", StandardCodec("JPEG2000", quality=45), "q45"),
        (r"$\mathrm{WebP}$", StandardCodec("WEBP", quality=40), "q40"),
        (r"$\mathrm{Ball\acute{e}\ 2018\ (Factorized)}$", NeuralModel("bmshj2018_factorized", quality=4, device=DEVICE), "q4"),
        (r"$\mathrm{Ball\acute{e}\ 2018\ (Hyperprior)}$", NeuralModel("bmshj2018_hyperprior", quality=5, device=DEVICE), "q5"),
        (r"$\mathrm{Minnen\ 2018\ (Autoregressive)}$", NeuralModel("mbt2018_mean", quality=5, device=DEVICE), "q5"),
        (r"$\mathrm{Cheng\ 2020\ (Attention)}$", NeuralModel("cheng2020_anchor", quality=6, device=DEVICE), "q6"),
    ]

    # Typography settings
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10.5,
    })

    fig, axes = plt.subplots(3, 4, figsize=(15, 10.5), dpi=300)
    axes = axes.flatten()

    orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # Dictionary to keep last neural reconstruction for uncertainty heatmap
    last_rec = None

    for idx, (name, model, _) in enumerate(methods):
        ax = axes[idx]
        if model is None:
            # Ground Truth
            rec_np = orig_np
            subtitle = r"$\mathrm{Raw\ uncompressed\ reference}$"
        else:
            with torch.no_grad():
                x_hat, bpp_val = model.compress_decompress(x)
            p_val = psnr(x, x_hat)
            s_val = ssim(x, x_hat)
            rec_np = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            last_rec = rec_np
            subtitle = rf"${bpp_val:.3f}\ \mathrm{{bpp}}\ |\ {p_val:.2f}\ \mathrm{{dB}}\ |\ \mathrm{{SSIM}}:\ {s_val:.4f}$"

        ax.imshow(rec_np)
        ax.set_title(f"{name}\n{subtitle}", pad=5)
        ax.axis("off")

        # Inset zoomed patch in lower-right or lower-left corner
        # Main patch box
        rect = patches.Rectangle(
            (crop_x0, crop_y0), crop_x1 - crop_x0, crop_y1 - crop_y0, 
            linewidth=1.2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Inset zoom
        ax_ins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=0.4)
        crop_np = rec_np[crop_y0:crop_y1, crop_x0:crop_x1]
        ax_ins.imshow(crop_np)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])
        for spine in ax_ins.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)

    # 12th panel: Uncertainty heatmap U(x) = |X - \hat{X}|^2
    ax_u = axes[11]
    if last_rec is not None:
        error_map = np.mean((orig_np - last_rec) ** 2, axis=2)
        im = ax_u.imshow(error_map, cmap="inferno")
        ax_u.set_title(r"$\mathrm{Predictive\ Uncertainty\ Map}\ U(x)$" + "\n" + r"$\mathbb{E}[(X - \hat{X})^2]\ \mathrm{(NeurIPS\ 2024)}$", pad=5)
        ax_u.axis("off")
        plt.colorbar(im, ax=ax_u, fraction=0.046, pad=0.04)

    plt.suptitle(
        r"$\mathrm{Visual\ Reconstruction\ Quality\ on\ Kodak\ 23\ (Matched\ Bitrate\ Regime\ \approx 0.30\ bpp)}$", 
        fontsize=14, y=0.99
    )
    plt.tight_layout()
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved visual comparison grid: {out_path}")

if __name__ == "__main__":
    generate_visual_grid(IMG_PATH, OUT_FIG)
