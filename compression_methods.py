"""
compression_methods.py - Clean and modular image compression implementations.
Each method is a standalone class providing:
    compress_decompress(x, noise_std=0.0) -> (x_hat, bpp)
    - Input: x (Tensor [1, C, H, W] in [0, 1])
    - Output: x_hat (Reconstructed Tensor [1, C, H, W]), bpp (float)
"""
import io
import torch
import torch.fft
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from PIL import Image
import compressai.zoo as zoo

def calculate_entropy_bpp(coefficients: np.ndarray, num_pixels: int) -> float:
    """Estimate bits per pixel (bpp) using empirical Shannon entropy of coefficients."""
    flat = coefficients.flatten()
    _, counts = np.unique(flat, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return float(entropy * len(flat) / num_pixels)

# =====================================================================
# 1. Classical Mathematical Transforms (DFT, DCT, Wavelet)
# =====================================================================
class DFTCompression:
    """Compression by selecting top-k% highest magnitude Fourier coefficients."""
    def __init__(self, keep_ratio: float = 0.10, q_step: float = 0.05):
        self.keep_ratio = keep_ratio
        self.q_step = q_step

    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        B, C, H, W = x.shape
        num_pixels = H * W
        X_fft = torch.fft.fft2(x, norm="ortho")
        magnitude = torch.abs(X_fft)

        # Retain top-k coefficients
        k = max(1, int(self.keep_ratio * H * W))
        threshold = torch.topk(magnitude.view(B, C, -1), k, dim=-1)[0][..., -1].view(B, C, 1, 1)
        mask = (magnitude >= threshold).float()

        # Frequency domain quantization
        X_kept = X_fft * mask
        real_q = torch.round(X_kept.real / self.q_step)
        imag_q = torch.round(X_kept.imag / self.q_step)

        # Rate estimation (coefficients entropy + sparsity mask overhead)
        bpp = calculate_entropy_bpp(real_q[mask > 0].cpu().numpy(), num_pixels) + \
              calculate_entropy_bpp(imag_q[mask > 0].cpu().numpy(), num_pixels) + \
              (k * np.log2(H * W) / num_pixels)

        # Dequantization
        X_recon = torch.complex(real_q * self.q_step, imag_q * self.q_step)

        # Injected representation perturbation for uncertainty evaluation
        if noise_std > 0.0:
            X_recon = X_recon + torch.randn_like(X_recon) * noise_std * mask

        x_hat = torch.fft.ifft2(X_recon, norm="ortho").real.clamp(0.0, 1.0)
        return x_hat, max(0.01, bpp)

class DCTCompression:
    """Standard 8x8 block-based Discrete Cosine Transform compression."""
    def __init__(self, quality: int = 50):
        self.quality = quality
        # Standard JPEG luminance quantization matrix
        base_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        scale = 50.0 / quality if quality < 50 else (100.0 - quality) / 50.0
        self.q_table = np.clip(base_table * scale, 1.0, 255.0)

    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        x_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        H, W, C = x_np.shape

        # Pad image boundaries to multiples of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_np = np.pad(x_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        pad_H, pad_W, _ = x_np.shape
        x_recon = np.zeros_like(x_np)
        all_quantized = []

        # Process 8x8 blocks independently
        for c in range(C):
            for i in range(0, pad_H, 8):
                for j in range(0, pad_W, 8):
                    block = (x_np[i:i+8, j:j+8, c] * 255.0) - 128.0
                    d = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    q = np.round(d / self.q_table)
                    all_quantized.append(q.flatten())

                    deq = q * self.q_table
                    if noise_std > 0.0:
                        deq += np.random.normal(0, noise_std * 20.0, deq.shape)

                    id_block = idct(idct(deq.T, norm='ortho').T, norm='ortho')
                    x_recon[i:i+8, j:j+8, c] = (id_block + 128.0) / 255.0

        x_hat = torch.from_numpy(x_recon[:H, :W, :]).permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0).to(x.device)
        bpp = calculate_entropy_bpp(np.concatenate(all_quantized), H * W)
        return x_hat, max(0.01, bpp)

class WaveletCompression:
    """Multilevel 2D Discrete Wavelet Transform (DWT) subband compression."""
    def __init__(self, wavelet: str = "db4", threshold: float = 0.05):
        self.wavelet = wavelet
        self.threshold = threshold

    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        x_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        H, W, C = x_np.shape
        x_recon = np.zeros_like(x_np)
        all_coeffs = []

        for c in range(C):
            coeffs = pywt.wavedec2(x_np[:, :, c], wavelet=self.wavelet, level=3)
            new_coeffs = [coeffs[0]]

            # Soft-threshold and quantize detail subbands
            for detail_tuple in coeffs[1:]:
                new_detail = []
                for subband in detail_tuple:
                    sub_thresh = pywt.threshold(subband, self.threshold, mode='soft')
                    q = np.round(sub_thresh / (self.threshold + 1e-6))
                    all_coeffs.append(q.flatten())

                    deq = q * (self.threshold + 1e-6)
                    if noise_std > 0.0:
                        deq += np.random.normal(0, noise_std * self.threshold, deq.shape)
                    new_detail.append(deq)
                new_coeffs.append(tuple(new_detail))

            rec_channel = pywt.waverec2(new_coeffs, wavelet=self.wavelet)
            x_recon[:, :, c] = rec_channel[:H, :W]

        x_hat = torch.from_numpy(x_recon).permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0).to(x.device)
        bpp = calculate_entropy_bpp(np.concatenate(all_coeffs), H * W)
        return x_hat, max(0.01, bpp)

# =====================================================================
# 2. Industry Standard Codecs (JPEG, WebP)
# =====================================================================
class StandardCodec:
    """Standard image compression wrapper (JPEG, WebP, JPEG2000 via Pillow)."""
    def __init__(self, fmt: str = "JPEG", quality: int = 75):
        self.fmt = fmt.upper()
        self.quality = quality

    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        x_np = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(x_np)
        H, W, _ = x_np.shape

        buffer = io.BytesIO()
        if self.fmt == "JPEG2000":
            # For JPEG 2000, quality_mode can be controlled via quality_layers
            # quality from 20 to 80 roughly translates to compression rates
            rate_factor = max(1, 100 - self.quality)
            img.save(buffer, format="JPEG2000", quality_layers=[self.quality])
        else:
            img.save(buffer, format=self.fmt, quality=self.quality)
            
        bpp = (buffer.tell() * 8.0) / (H * W)

        buffer.seek(0)
        dec_img = Image.open(buffer).convert("RGB")
        dec_np = np.array(dec_img, dtype=np.float32) / 255.0
        x_hat = torch.from_numpy(dec_np).permute(2, 0, 1).unsqueeze(0).float()

        if noise_std > 0.0:
            x_hat += torch.randn_like(x_hat) * noise_std

        return x_hat.clamp(0.0, 1.0).to(x.device), float(bpp)

# =====================================================================
# 3. Deep Learning Neural Codecs (CompressAI Zoo)
# =====================================================================
class NeuralModel:
    """Pretrained end-to-end neural image codecs from CompressAI."""
    def __init__(self, model_name: str = "bmshj2018_hyperprior", quality: int = 3, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        builder = getattr(zoo, model_name)
        self.net = builder(quality=quality, pretrained=True).eval().to(self.device)

    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor, noise_std: float = 0.0):
        x = x.to(self.device)
        _, _, h, w = x.shape

        # Pad to multiple of 64
        nh, nw = (h + 63) // 64 * 64, (w + 63) // 64 * 64
        x_pad = torch.nn.functional.pad(x, (0, nw - w, 0, nh - h), mode="replicate")

        out = self.net(x_pad)
        x_hat = out["x_hat"][:, :, :h, :w].clamp(0.0, 1.0)

        # Rate computation from likelihoods
        total_bpp = 0.0
        for _, v in out["likelihoods"].items():
            branch_bpp = torch.log(v.clamp(min=1e-9)).sum() / (-torch.log(torch.tensor(2.0)) * h * w)
            total_bpp += float(branch_bpp.item())

        if noise_std > 0.0:
            x_hat = (x_hat + torch.randn_like(x_hat) * noise_std).clamp(0.0, 1.0)

        return x_hat, total_bpp
