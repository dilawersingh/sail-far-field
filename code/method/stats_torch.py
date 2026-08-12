import torch
import os
import numpy as np
from PIL import Image

def normalize_intensity_sum(I: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Faithful mirror of stats.normalize_intensity(..., mode="sum"):

    - clamps I to >= 0
    - divides by per-sample sum over last two dims (H,W) with keepdim=True

    I: (..., H, W)  (typically (B,H,W))
    returns: same shape as I
    """
    if I.ndim < 2:
        raise ValueError(f"normalize_intensity_sum expected at least 2 dims (...,H,W), got {I.shape}")

    I = I.clamp_min(0.0)
    denom = I.sum(dim=(-2, -1), keepdim=True) + eps
    return I / denom

def field_to_phase(Y, eps=1e-8):
    """
    Y: (B,2,H,W) tensor, interpreted as real/imag channels.
    Returns:
        phase: (B,H,W) in [-pi, pi]
    """
    real = Y[:, 0]
    imag = Y[:, 1]
    mag = torch.sqrt(real**2 + imag**2 + eps)
    real_u = real / mag
    imag_u = imag / mag
    phase = torch.atan2(imag_u, real_u)
    return phase

def phase_to_uint8(phase):
    """
    phase: (...,H,W) tensor or ndarray in [-pi, pi]
    Returns uint8 in [0,255]
    """
    if isinstance(phase, torch.Tensor):
        phase = phase.detach().cpu().numpy()
    phase_wrapped = (phase + np.pi) / (2 * np.pi)   # [0,1]
    phase_8bit = np.clip(np.round(255 * phase_wrapped), 0, 255).astype(np.uint8)
    return phase_8bit

def phase_to_field(phase):
    """
    phase: (B,H,W)
    returns: (B,2,H,W)
    """
    real = torch.cos(phase)
    imag = torch.sin(phase)
    return torch.stack([real, imag], dim=1)

def save_phase_outputs(Y, out_dir, prefix="best"):
    """
    Saves:
      - full precision phase .npy
      - 8-bit phase .png
    Uses first batch element.
    """
    os.makedirs(out_dir, exist_ok=True)

    phase = field_to_phase(Y)[0]  # (H,W)
    phase_np = phase.detach().cpu().numpy()
    phase_8bit = phase_to_uint8(phase_np)

    np.save(os.path.join(out_dir, f"{prefix}_phase.npy"), phase_np)
    Image.fromarray(phase_8bit).save(os.path.join(out_dir, f"{prefix}_phase_8bit.png"))

    return phase_np, phase_8bit