# Uncertainty & Lossy Image Compression Benchmark

A lightweight, self-contained research repository for evaluating lossy image compression methods (Classical Transforms, Standard Codecs, and Neural AI Codecs) on the **Kodak** true color suite.

Everything is included inside this folder: dataset, reference papers, compression implementations, and automated evaluation.

---

## 📁 Repository Structure

```
uncertainty/
├── data/                    # 24 Kodak benchmark images (PNG)
├── papers/                  # Foundational reference research papers (PDF)
├── compression_methods.py   # Standalone compression methods (DFT, DCT, Wavelet, JPEG, WebP, Neural)
├── metrics.py               # Evaluation metrics: bpp (Rate), PSNR (Distortion), SSIM, LPIPS, Uncertainty (U)
├── run_benchmark.py         # Self-contained script: inference, evaluation, summary table, and LaTeX plot
├── results/                 # Output benchmark CSV and publication figures
└── README.md                # Documentation
```

---

## 🚀 Instant Run

Clone the repository and run immediately without setting external dataset paths:

```bash
cd uncertainty
python run_benchmark.py
```

The script will automatically:
1. Load images from `./data/kodak/`.
2. Run inference across all configured compression methods.
3. Compute Rate (bpp), Distortion (PSNR), SSIM, and LPIPS.
4. Output a clean summary table to stdout and save results to `results/benchmark.csv`.
5. Generate a publication-quality LaTeX figure `results/rd_plot.png` with non-overlapping method labels.

---

## 🛠️ How to Add a New Compression Method

### Step 1: Define your method in `compression_methods.py`
Any new method only needs a `compress_decompress(x)` method:

```python
class MyCustomCompression:
    def __init__(self, param=1.0):
        self.param = param

    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        # x: Input tensor [1, 3, H, W] in [0, 1]
        x_hat = x # Your compression/reconstruction logic
        bpp = 0.35 # Calculated or estimated bit rate
        return x_hat, bpp
```

### Step 2: Register it in `run_benchmark.py`
Add one line to the `METHODS` dictionary:
```python
METHODS = {
    ...
    r"$\mathrm{My\ Method}$": MyCustomCompression(param=2.0),
}
```
Re-run `python run_benchmark.py` and your method will automatically be evaluated and plotted!
