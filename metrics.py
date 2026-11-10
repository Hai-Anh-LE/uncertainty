"""
metrics.py - Evaluation metrics for Rate, Distortion, Perception, and Uncertainty.
All implementations are fully vectorised and reproducible.
"""
import torch
import torch.nn.functional as F
import lpips
from pytorch_msssim import ms_ssim as compute_ms_ssim

_lpips_fn = None

def get_lpips(device: str = "cpu"):
    """Singleton getter for AlexNet LPIPS perceptual loss model."""
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _lpips_fn

def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR in dB) for images normalized to [0, 1]."""
    mse = F.mse_loss(x, y).clamp(min=1e-10)
    return float((10.0 * torch.log10(1.0 / mse)).item())

def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Multi-Scale Structural Similarity Index (MS-SSIM)."""
    try:
        return float(compute_ms_ssim(x, y, data_range=1.0).item())
    except Exception:
        return float("nan")

def lpips_score(x: torch.Tensor, y: torch.Tensor, device: str = "cuda") -> float:
    """Compute perceptual distance score using AlexNet backbone."""
    try:
        fn = get_lpips(device)
        score = fn((x.to(device) * 2.0) - 1.0, (y.to(device) * 2.0) - 1.0)
        return float(score.mean().item())
    except Exception:
        return float("nan")

def uncertainty_u(samples_tensor: torch.Tensor):
    """
    Compute predictive uncertainty U = E[Var(X_hat | X)] from K stochastic decodings.
    samples_tensor shape: [K, 1, C, H, W]
    Returns:
        scalar_u: Average conditional variance across all pixels and channels.
        var_map: Per-pixel variance map of shape [H, W].
    """
    var_map = torch.var(samples_tensor, dim=0, unbiased=True).mean(dim=1).squeeze(0)
    scalar_u = float(var_map.mean().item())
    return scalar_u, var_map
