import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt

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
    exclude_mask=None,
    eps: float = 1e-12,
    scale_to_numel: bool = True,
) -> torch.Tensor:
    if I.ndim < 2:
        raise ValueError(f"Expected at least 2 dims (...,H,W), got {I.shape}")

    I = I.clamp_min(0.0)

    if exclude_mask is None:
        denom = I.sum(dim=(-2, -1), keepdim=True) + eps
        out = I / denom
    else:
        if exclude_mask.dtype != torch.bool:
            exclude_mask = exclude_mask.bool()

        while exclude_mask.ndim < I.ndim:
            exclude_mask = exclude_mask.unsqueeze(0)

        include = (~exclude_mask).to(dtype=I.dtype, device=I.device)
        denom = (I * include).sum(dim=(-2, -1), keepdim=True) + eps
        out = I / denom

    if scale_to_numel:
        H, W = I.shape[-2], I.shape[-1]
        out = out * (H * W)

    return out

def checkerboard_overlay(a, b, tile=40):
    H, W = a.shape
    out = np.zeros_like(a)

    for y in range(0, H, tile):
        for x in range(0, W, tile):
            if ((x // tile + y // tile) % 2) == 0:
                out[y:y+tile, x:x+tile] = a[y:y+tile, x:x+tile]
            else:
                out[y:y+tile, x:x+tile] = b[y:y+tile, x:x+tile]

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
    dc_center=None,
    subtract_min=True,
    zero_dc_for_loss=True,
    median_ksize=0,
):
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=np.float32)

    if angle:
        gray = rotate_image(gray, angle)

    y0 = roi["y0"]
    x0 = roi["x0"]
    h = roi["h"]
    w = roi["w"]

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

    I_cam = normalize_intensity_sum_excluding_mask(
        I_cam_raw,
        exclude_mask=dc_mask_t,
        eps=eps_norm,
        scale_to_numel=True,
    )

    if zero_dc_for_loss:
        I_cam = I_cam.masked_fill(dc_mask_t.unsqueeze(0), 0.0)

    return I_cam, dc_mask_t, crop_resized, (cx, cy)

def normalize01(im, eps=1e-8):
    im = im.astype(np.float32)
    im = im - im.min()
    return im / (im.max() + eps)

# -----------------------------
# Inputs
# -----------------------------

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
image_path = str(sorted(paths.TARGETS.glob("*.png"))[0])  # pick any
target_path = image_path

CAMERA_ROI = {
    "y0": 18,
    "x0": 750,
    "h": 2481,
    "w": 2481
}
  
DC_RADIUS = 20
DC_AUTO_CENTER = False
DC_CENTER = (500, 500)
DC_ZERO_IN_LOSS = False
ROTATE_DEGREES = 0.3
EPS_NORM = 1e-12

# CAMERA_ROI = { #Bottom Left Quadrant
#     "y0": 750,
#     "x0": 0,
#     "h": 1838,
#     "w": 1838
# }

# DC_RADIUS = 20
# DC_AUTO_CENTER = False
# DC_CENTER = (500, 500)
# DC_ZERO_IN_LOSS = False
# ROTATE_DEGREES = 0.2

device = "cpu"
eps_norm = 1e-12
out_hw = (1000, 1000)

# -----------------------------
# Load original image
# -----------------------------
img = Image.open(image_path).convert("L")
X = np.asarray(img, dtype=np.float32)

if ROTATE_DEGREES:
    X_rot = rotate_image(X, ROTATE_DEGREES)
else:
    X_rot = X.copy()

# raw crop before resize
y0 = CAMERA_ROI["y0"]
x0 = CAMERA_ROI["x0"]
h = CAMERA_ROI["h"]
w = CAMERA_ROI["w"]
crop_raw = X_rot[y0:y0+h, x0:x0+w]

# processed camera version using YOUR trusted pipeline
I_cam, dc_mask_t, cam_resized_np, dc_center_used = load_camera_capture_for_citl(
    image_path=image_path,
    angle=ROTATE_DEGREES,
    roi=CAMERA_ROI,
    out_hw=out_hw,
    device=device,
    eps_norm=eps_norm,
    dc_radius=DC_RADIUS,
    auto_center=DC_AUTO_CENTER,
    dc_center=DC_CENTER,
    subtract_min=True,
    zero_dc_for_loss=DC_ZERO_IN_LOSS,
    median_ksize=0,
)

cx, cy = dc_center_used

# -----------------------------
# Load target and resize if needed
# -----------------------------
target = Image.open(target_path).convert("L")
target_np = np.asarray(target, dtype=np.float32)

if target_np.shape != out_hw:
    target_np = cv2.resize(target_np, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

# normalize only for display/comparison
target_vis = normalize01(target_np)
cam_vis = normalize01(cam_resized_np)

# optional DC-masked camera for comparison
cam_vis_masked = cam_vis.copy()
cam_vis_masked[dc_mask_t.cpu().numpy()] = 0.0

diff = target_vis - cam_vis
diff_masked = target_vis - cam_vis_masked

mse = np.mean(diff**2)
mae = np.mean(np.abs(diff))

mse_masked = np.mean(diff_masked**2)
mae_masked = np.mean(np.abs(diff_masked))

print(f"Before masking DC:  MAE={mae:.6f}, MSE={mse:.6f}")
print(f"After masking DC:   MAE={mae_masked:.6f}, MSE={mse_masked:.6f}")

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

# 1) original with ROI box
ax = axes[0, 0]
ax.imshow(X_rot, cmap="gray")
rect = plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor="none", linewidth=1)
ax.add_patch(rect)
ax.set_title("Original (with ROI)")
ax.axis("off")

# 2) cropped ROI
ax = axes[0, 1]
ax.imshow(crop_raw, cmap="gray")
ax.set_title(f"Cropped ROI ({crop_raw.shape[0]} x {crop_raw.shape[1]})")
ax.axis("off")

# 3) resized crop with DC mask overlay
ax = axes[0, 2]
ax.imshow(cam_resized_np, cmap="gray")
circle = plt.Circle((cx, cy), DC_RADIUS, edgecolor="lime", facecolor="none", linewidth=2)
ax.add_patch(circle)
ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
ax.set_title(f"Resized Camera\ncenter=({cx}, {cy}), r={DC_RADIUS}")
ax.axis("off")

# 4) target
ax = axes[1, 0]
ax.imshow(target_vis, cmap="gray")
ax.set_title("Target")
ax.axis("off")

# 5) overlay target vs processed camera
ax = axes[1, 1]
overlay = np.stack([target_vis, cam_vis, np.zeros_like(target_vis)], axis=-1)
ax.imshow(overlay)
ax.set_title("Overlay\nTarget=R, Camera=G")
ax.axis("off")

# 6) difference map
ax = axes[1, 2]
im = ax.imshow(diff_masked, cmap="bwr", vmin=-1, vmax=1)
ax.set_title(f"Difference (Target - Camera)\nmasked DC | MAE={mae_masked:.4f}, MSE={mse_masked:.4f}")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.show()

checker = checkerboard_overlay(target_vis, cam_vis_masked, tile=40)

plt.figure(figsize=(6,6))
plt.imshow(checker, cmap="gray")
plt.title("Checkerboard Overlay")
plt.axis("off")
plt.show()

target_u8 = (255 * target_vis).astype(np.uint8)
cam_u8 = (255 * cam_vis).astype(np.uint8)

edges_t = cv2.Canny(target_u8, 50, 150)
edges_c = cv2.Canny(cam_u8, 50, 150)

overlay = np.zeros((target_vis.shape[0], target_vis.shape[1], 3), dtype=np.float32)
overlay[..., 0] = edges_t / 255.0   # red = target edges
overlay[..., 1] = edges_c / 255.0   # green = camera edges

plt.figure(figsize=(7, 7))
plt.imshow(overlay)
plt.title("Edge Overlay: target edges = red, camera edges = green")
plt.axis("off")
plt.show()