"""
plot_rd.py - Rate-Distortion curves exactly matching the CompressAI official benchmark style.
- Ascending solid curves from left to right.
- Points are colored solid dots matching their respective curve colors (no stars, no gray circles).
- Axis limits set to [0.0, 2.0] bpp and [26.0, 42.0] dB.
- LaTeX Computer Modern typography.
"""
import os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "results/benchmark.csv"
OUT_FIG = REPO_ROOT / "results/rd_plot.png"

# Color mapping inspired by CompressAI official plots
COLOR_PALETTE = {
    r"$\mathrm{Cheng\ 2020\ (Attention)}$": "#ff7f0e",     # Orange (top performer)
    r"$\mathrm{Minnen\ 2018\ (Autoregressive)}$": "#2ca02c",# Green
    r"$\mathrm{Ball\acute{e}\ 2018\ (Hyperprior)}$": "#9467bd", # Purple
    r"$\mathrm{Ball\acute{e}\ 2018\ (Factorized)}$": "#bcbd22", # Olive
    r"$\mathrm{WebP}$": "#17becf",                           # Cyan
    r"$\mathrm{JPEG\ 2000}$": "#1f77b4",                     # Blue
    r"$\mathrm{JPEG}$": "#e377c2",                          # Pink/Salmon
    r"$\mathrm{DCT\ (Block\ 8\times 8)}$": "#8c564b",       # Brown
    r"$\mathrm{DWT\ (Wavelet\ db4)}$": "#d62728",           # Red
}

def plot_rd(csv_path: pathlib.Path, out_path: pathlib.Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    group_col = "family" if "family" in df.columns else "method"
    agg_cols = [group_col, "q_tag"] if "q_tag" in df.columns else [group_col]
    summary = df.groupby(agg_cols)[["bpp", "psnr", "ssim", "lpips"]].mean().reset_index()

    # LaTeX Typography settings
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # Define display order matching CompressAI hierarchy (top performer to lowest)
    legend_order = [
        r"$\mathrm{Cheng\ 2020\ (Attention)}$",
        r"$\mathrm{Minnen\ 2018\ (Autoregressive)}$",
        r"$\mathrm{Ball\acute{e}\ 2018\ (Hyperprior)}$",
        r"$\mathrm{Ball\acute{e}\ 2018\ (Factorized)}$",
        r"$\mathrm{WebP}$",
        r"$\mathrm{JPEG\ 2000}$",
        r"$\mathrm{JPEG}$",
        r"$\mathrm{DCT\ (Block\ 8\times 8)}$",
        r"$\mathrm{DWT\ (Wavelet\ db4)}$",
    ]
    present_families = [f for f in legend_order if f in summary[group_col].unique()]
    # Any extra families not in legend_order
    for f in summary[group_col].unique():
        if f not in present_families:
            present_families.append(f)

    for idx, fam_name in enumerate(present_families):
        fam_data = summary[summary[group_col] == fam_name].sort_values(by="bpp")
        
        # Filter points within display range [0, 2.2] bpp
        fam_data = fam_data[fam_data["bpp"] <= 2.2]
        if len(fam_data) == 0:
            continue

        c = COLOR_PALETTE.get(fam_name, plt.cm.tab10(idx % 10))

        # Plot solid line with matching colored solid circular markers
        ax.plot(
            fam_data["bpp"], 
            fam_data["psnr"], 
            linestyle="-", 
            linewidth=1.8, 
            marker="o", 
            markersize=5, 
            markerfacecolor=c, 
            markeredgecolor=c, 
            color=c, 
            label=fam_name
        )

    # CompressAI Axis ranges and ticks
    ax.set_xlim(0.0, 2.2)
    ax.set_ylim(26.0, 42.0)
    ax.set_xticks(np.arange(0.0, 2.25, 0.25))
    ax.set_yticks(np.arange(26.0, 43.0, 2.0))

    ax.set_xlabel(r"$\mathrm{Bit\text{-}rate\ [bpp]}$")
    ax.set_ylabel(r"$\mathrm{PSNR\ [dB]}$")
    ax.set_title(r"$\mathrm{Performance\ evaluation\ on\ Kodak\ -\ PSNR\ (RGB)}$")
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.6)
    
    # Legend box placed in upper right with 2 columns to prevent overlapping ANY curve (especially high-bitrate JPEG)
    ax.legend(
        frameon=True, 
        loc="upper right", 
        ncol=2, 
        fontsize=8.8, 
        columnspacing=1.0, 
        handletextpad=0.4,
        framealpha=0.95, 
        edgecolor="gray"
    )

    plt.tight_layout()
    os.makedirs(out_path.parent, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Saved CompressAI-style plot: {out_path}")

if __name__ == "__main__":
    plot_rd(CSV_PATH, OUT_FIG)
