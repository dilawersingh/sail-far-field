from datetime import datetime
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from stats_torch import normalize_intensity_sum
import cv2


# ---------------------------
# User settings
# ---------------------------
# Defaults for running this file as a standalone script. The deposited
# analysis imports the functions below and never reads these two values.
INPUT_PATH = "path/to/capture.CR2"  # can also be .NEF, .ARW, .DNG, .png, .tif, .jpg
OUTPUT_DIR = "path/to/output"

# -----------------------------
# User-editable hardware settings
# -----------------------------
SLM_SHAPE = (1080, 1920)

# -----------------------------
# Helpers
# -----------------------------
def rename_capture(result_path, capture_dir, stem):
    """
    Rename the downloaded capture to a controlled filename.
    
    Parameters
    ----------
    result_path : str or Path
        Original path returned by camera.capture_image()
    capture_dir : str or Path
        Directory where the renamed file should live
    stem : str
        Desired filename stem without extension

    Returns
    -------
    Path
        New renamed file path
    """
    result_path = Path(result_path)
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    suffix = result_path.suffix.lower()
    new_path = capture_dir / f"{stem}{suffix}"

    # Avoid accidental overwrite
    if new_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_path = capture_dir / f"{stem}_{timestamp}{suffix}"

    result_path.rename(new_path)
    return new_path

def rotate_image(gray, angle_deg):
    h, w = gray.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated = cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated


def circular_mask_np(shape, cx, cy, radius):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2


def find_dc_center_np(gray):
    y, x = np.unravel_index(np.argmax(gray), gray.shape)
    return int(x), int(y)


def normalize_intensity_sum_excluding_mask(
    I: torch.Tensor,
    exclude_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    scale_to_numel: bool = True,
) -> torch.Tensor:
    """
    Sum-normalize I while excluding masked pixels from the denominator.

    I: (..., H, W)
    exclude_mask: broadcastable boolean mask with shape (H,W) or (1,H,W) or (...,H,W)
                  True means EXCLUDE from normalization sum.
    Returns tensor with same shape as I.

    If scale_to_numel=True, multiply result by H*W after normalization,
    matching your existing target scaling.
    """
    if I.ndim < 2:
        raise ValueError(f"Expected at least 2 dims (...,H,W), got {I.shape}")

    I = I.clamp_min(0.0)

    if exclude_mask is None:
        denom = I.sum(dim=(-2, -1), keepdim=True) + eps
        out = I / denom
    else:
        if exclude_mask.dtype != torch.bool:
            exclude_mask = exclude_mask.bool()

        # Broadcast mask to I shape
        while exclude_mask.ndim < I.ndim:
            exclude_mask = exclude_mask.unsqueeze(0)

        include = (~exclude_mask).to(dtype=I.dtype, device=I.device)
        denom = (I * include).sum(dim=(-2, -1), keepdim=True) + eps
        out = I / denom

    if scale_to_numel:
        H, W = I.shape[-2], I.shape[-1]
        out = out * (H * W)

    return out

def load_camera_capture_for_citl(
    image_path,
    roi,
    out_hw,
    device,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,      # tuple (cx, cy) in resized-image coordinates if auto_center=False
    subtract_min=True,
    median_ksize=0,
):
    """
    Load grayscale DSLR capture, crop ROI, resize to out_hw, and normalize
    while excluding the DC region from the normalization denominator.

    Returns:
        I_cam              : (1,H,W) float tensor, normalized for loss
        cam_np_resized     : resized raw grayscale numpy image (for saving/display)
        dc_mask_t          : (H,W) bool tensor on device
        dc_center_used     : (cx, cy)
    """
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=np.float32)

    y0 = roi["y0"]
    x0 = roi["x0"]
    h = roi["h"]
    w = roi["w"]

    if angle:
        gray = rotate_image(gray, angle)

    crop = gray[y0:y0+h, x0:x0+w]
    if crop.shape != (h, w):
        raise ValueError(
            f"Requested ROI {(h, w)} at {(y0, x0)} is out of bounds for capture shape {gray.shape}"
        )

    if median_ksize and median_ksize >= 3:
        if median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd")
        crop = cv2.medianBlur(crop.astype(np.float32), median_ksize)

    H, W = out_hw
    crop_resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if subtract_min:
        crop_resized = crop_resized - crop_resized.min()
        crop_resized = np.clip(crop_resized, 0.0, None)

    if auto_center:
        cx, cy = find_dc_center_np(crop_resized)
    else:
        if dc_center is None:
            raise ValueError("dc_center must be provided when auto_center=False")
        cx, cy = int(dc_center[0]), int(dc_center[1])

    dc_mask_np = circular_mask_np((H, W), cx, cy, dc_radius)

    I_cam_raw = torch.from_numpy(crop_resized).unsqueeze(0).to(device=device, dtype=torch.float32)
    dc_mask_t = torch.from_numpy(dc_mask_np).to(device=device, dtype=torch.bool)

    # Normalize excluding DC, but do NOT zero it in the saved/displayed image
    I_cam = normalize_intensity_sum_excluding_mask(
        I_cam_raw,
        exclude_mask=dc_mask_t,
        eps=eps_norm,
        scale_to_numel=True,
    )

    return I_cam, crop_resized, dc_mask_t, (cx, cy)

def no_dc_load_camera_capture_for_citl(
    image_path,
    roi,
    out_hw,
    device,
    angle=0,
    subtract_min=True,
    eps_norm=1e-12,
    median_ksize=0,
):
    """
    Load grayscale DSLR capture, crop ROI, resize to out_hw.

    Returns:
        I_cam          : (1,H,W) float tensor
        cam_np_resized : resized raw grayscale numpy image
    """
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=np.float32)

    y0 = roi["y0"]
    x0 = roi["x0"]
    h = roi["h"]
    w = roi["w"]

    if angle:
        gray = rotate_image(gray, angle)

    crop = gray[y0:y0+h, x0:x0+w]
    if crop.shape != (h, w):
        raise ValueError(
            f"Requested ROI {(h, w)} at {(y0, x0)} is out of bounds for capture shape {gray.shape}"
        )

    if median_ksize and median_ksize >= 3:
        if median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd")
        crop = cv2.medianBlur(crop.astype(np.float32), median_ksize)

    H, W = out_hw
    crop_resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if subtract_min:
        crop_resized = crop_resized - crop_resized.min()
        crop_resized = np.clip(crop_resized, 0.0, None)

    I_cam_raw = torch.from_numpy(crop_resized).unsqueeze(0).to(device=device, dtype=torch.float32)
    I_cam = normalize_intensity_sum(I_cam_raw, eps=eps_norm) * (H * W)

    return I_cam, crop_resized