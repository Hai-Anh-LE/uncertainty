"""
run_benchmark.py - Evaluates rate sweeps (q1..q6) across all compression families
to plot continuous Rate-Distortion curves (bpp vs PSNR) matching CompressAI standards.
"""
import os
import pathlib
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from compression_methods import (
    DFTCompression, 
    DCTCompression, 
    WaveletCompression, 
    StandardCodec, 
    NeuralModel
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = pathlib.Path(__file__).resolve().parent
KODAK_DIR = REPO_ROOT / "data/kodak"

# =====================================================================
# Compression families with multiple quality levels (Sweep)
# Each family produces a continuous RD curve from left to right!
# =====================================================================
FAMILIES = {
    # 1. Classical Transforms
    r"$\mathrm{DCT\ (Block\ 8\times 8)}$": [
        ("q10", DCTCompression(quality=10)),
        ("q20", DCTCompression(quality=20)),
        ("q40", DCTCompression(quality=40)),
        ("q60", DCTCompression(quality=60)),
        ("q80", DCTCompression(quality=80)),
    ],
    r"$\mathrm{DWT\ (Wavelet\ db4)}$": [
        ("th0.15", WaveletCompression(threshold=0.15)),
        ("th0.08", WaveletCompression(threshold=0.08)),
        ("th0.04", WaveletCompression(threshold=0.04)),
        ("th0.02", WaveletCompression(threshold=0.02)),
        ("th0.01", WaveletCompression(threshold=0.01)),
    ],

    # 2. Standard Formats
    r"$\mathrm{JPEG}$": [
        ("q10", StandardCodec("JPEG", quality=10)),
        ("q20", StandardCodec("JPEG", quality=20)),
        ("q40", StandardCodec("JPEG", quality=40)),
        ("q60", StandardCodec("JPEG", quality=60)),
        ("q80", StandardCodec("JPEG", quality=80)),
    ],
    r"$\mathrm{JPEG\ 2000}$": [
        ("q25", StandardCodec("JPEG2000", quality=25)),
        ("q30", StandardCodec("JPEG2000", quality=30)),
        ("q35", StandardCodec("JPEG2000", quality=35)),
        ("q40", StandardCodec("JPEG2000", quality=40)),
        ("q45", StandardCodec("JPEG2000", quality=45)),
    ],
    r"$\mathrm{WebP}$": [
        ("q10", StandardCodec("WEBP", quality=10)),
        ("q20", StandardCodec("WEBP", quality=20)),
        ("q40", StandardCodec("WEBP", quality=40)),
        ("q60", StandardCodec("WEBP", quality=60)),
        ("q80", StandardCodec("WEBP", quality=80)),
    ],

    # 3. CompressAI Neural Models (Full Sweeps matching CompressAI up to q6)
    r"$\mathrm{Ball\acute{e}\ 2018\ (Factorized)}$": [
        ("q1", NeuralModel("bmshj2018_factorized", quality=1, device=DEVICE)),
        ("q2", NeuralModel("bmshj2018_factorized", quality=2, device=DEVICE)),
        ("q3", NeuralModel("bmshj2018_factorized", quality=3, device=DEVICE)),
        ("q4", NeuralModel("bmshj2018_factorized", quality=4, device=DEVICE)),
        ("q5", NeuralModel("bmshj2018_factorized", quality=5, device=DEVICE)),
        ("q6", NeuralModel("bmshj2018_factorized", quality=6, device=DEVICE)),
    ],
    r"$\mathrm{Ball\acute{e}\ 2018\ (Hyperprior)}$": [
        ("q1", NeuralModel("bmshj2018_hyperprior", quality=1, device=DEVICE)),
        ("q2", NeuralModel("bmshj2018_hyperprior", quality=2, device=DEVICE)),
        ("q3", NeuralModel("bmshj2018_hyperprior", quality=3, device=DEVICE)),
        ("q4", NeuralModel("bmshj2018_hyperprior", quality=4, device=DEVICE)),
        ("q5", NeuralModel("bmshj2018_hyperprior", quality=5, device=DEVICE)),
        ("q6", NeuralModel("bmshj2018_hyperprior", quality=6, device=DEVICE)),
    ],
    r"$\mathrm{Minnen\ 2018\ (Autoregressive)}$": [
        ("q1", NeuralModel("mbt2018_mean", quality=1, device=DEVICE)),
        ("q2", NeuralModel("mbt2018_mean", quality=2, device=DEVICE)),
        ("q3", NeuralModel("mbt2018_mean", quality=3, device=DEVICE)),
        ("q4", NeuralModel("mbt2018_mean", quality=4, device=DEVICE)),
        ("q5", NeuralModel("mbt2018_mean", quality=5, device=DEVICE)),
        ("q6", NeuralModel("mbt2018_mean", quality=6, device=DEVICE)),
    ],
    r"$\mathrm{Cheng\ 2020\ (Attention)}$": [
        ("q1", NeuralModel("cheng2020_anchor", quality=1, device=DEVICE)),
        ("q2", NeuralModel("cheng2020_anchor", quality=2, device=DEVICE)),
        ("q3", NeuralModel("cheng2020_anchor", quality=3, device=DEVICE)),
        ("q4", NeuralModel("cheng2020_anchor", quality=4, device=DEVICE)),
        ("q5", NeuralModel("cheng2020_anchor", quality=5, device=DEVICE)),
        ("q6", NeuralModel("cheng2020_anchor", quality=6, device=DEVICE)),
    ],
}

def load_img(path: pathlib.Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

from metrics import psnr, ssim, lpips_score

def main():
    kodak_images = sorted(list(pathlib.Path(KODAK_DIR).glob("*.png")))
    print(f"[*] Running full curve sweep on {len(kodak_images)} Kodak images using {DEVICE.upper()}...")

    records = []
    for family_name, sweep in FAMILIES.items():
        print(f"\n--> Evaluating Family: {family_name}")
        for q_tag, model in sweep:
            for p in tqdm(kodak_images, desc=f"{family_name} ({q_tag})", leave=False):
                x = load_img(p)
                with torch.no_grad():
                    x_hat, bpp = model.compress_decompress(x)

                records.append({
                    "family": family_name,
                    "q_tag": q_tag,
                    "image": p.name,
                    "bpp": bpp,
                    "psnr": psnr(x, x_hat),
                    "ssim": ssim(x, x_hat),
                    "lpips": lpips_score(x, x_hat, device=DEVICE)
                })

    df = pd.DataFrame(records)
    os.makedirs(REPO_ROOT / "results", exist_ok=True)
    csv_path = REPO_ROOT / "results/benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved complete sweep results to: {csv_path}")

    # Automatically invoke plot_rd to generate curves
    from plot_rd import plot_rd
    plot_rd(csv_path, REPO_ROOT / "results/rd_plot.png")

if __name__ == "__main__":
    main()
