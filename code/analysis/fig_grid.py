r"""
fig_grid.py -- E8. Qualitative grid: every target, key methods, PSNR on every
panel.

Every capture panel here carries its PSNR, drawn from the canonical
dataset, so the qualitative and quantitative claims can never diverge: the
number printed on a panel IS the number in sail_scored.json for that
(physics, method, target) cell, by construction.

WHAT IT SHOWS (redesigned 2026-08-04). Per target and physics: every panel
saved individually under e8_panels/ (pick-and-choose for manuscript and
supplement, GD+CITL fig_reconstructions pattern), plus an assembled figure
under e8_grid/ with the target at left and the six COLUMNS methods as a 2x3
grid, PSNR badge on each capture. FNO excluded: purely experimental panel.
All captures come from the newest replay_converged run, one session, one
alignment, so differences between panels are differences between methods.

PROCESSING (native resolution, 2026-08-04, after Dilawer rejected the
downsampled tiles). Captures are loaded through the SAME loader the scoring
uses (load_camera_capture_for_citl) at the native 1000x1000 grid, so what the
figure shows is what the metric scored. Shown as captured: the zero order is
included, and the display scale (joint 99.7th percentile over the methods,
DC region excluded) clips the spike to saturated white rather than letting it
set the scale -- the GD+CITL convention exactly. The target keeps its own
scale. ROI crops are native pixels rendered with nearest interpolation; full
frames use antialiased. Render dpi is chosen so each panel keeps ~1000 px.

FILE NAMES VS METHOD NAMES. The manuscript's bench domain renames the
replay's gs_750/gd_750 to gs/gd (see sailrev). Files on disk keep the raw
names, so FILE_FOR maps back.

RUNTIME. ~220 DSLR frames are loaded and rotated at full resolution; expect a
few minutes. The notebook runs it once per rebuild, which is the point.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import sailrev as S

# Column order: target first, then methods, worst-to-best left-to-right so the
# eye reads improvement rightward. Keys are the manuscript's bench names.
COLUMNS = ["gs", "gd", "transformer_per_target", "gd_citl_warm",
           "sail", "batched_sail_2000"]
FILE_FOR = {"gs": "gs_750", "gd": "gd_750"}   # disk names in the replay run
# Panel-title-only abbreviations; the manuscript's full display names (sailrev
# LABEL) stay untouched everywhere else.
# Panel titles only; captions state that the CITL arms shown are the
# simulation-seeded ones and that Batched SAIL is the 2000-epoch budget.
SHORT = {"gd_citl_warm": "GD+CITL",
         "gs_citl_warm": "GS+CITL",
         "batched_sail_2000": "Batched SAIL",
         "batched_sail_750": "Batched SAIL"}
ROI_BOX_COLOUR = "#2ee06a"    # same green as the GD+CITL figures
ROI_FILENAME = "roi.json"     # {target: [y0_frac, x0_frac, side_frac]}

# Display grid is the native scoring grid; no TILE downsampling anywhere.
NATIVE_HW = (1000, 1000)
DC_CENTER_NATIVE = (495, 510)
DC_RADIUS_NATIVE = 20
TARGET_PERCENTILE = 99.9
CAPTURE_PERCENTILE = 99.7


def _capture_run() -> Path:
    root = (Path(os.environ["SAILREV_RESULTS"]) / "Self-Attention" /
            "multilevel" / "experiments" / "replay_converged")
    runs = sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("replay_converged_"))
    if not runs:
        raise FileNotFoundError(f"no replay_converged run under {root}")
    return runs[-1]


def _calibration(run: Path):
    import json
    cal = json.loads((run / "calibration.json").read_text())
    return cal["roi"], cal["rotation_deg"]


def _load_cam(path: Path, rig_roi: dict, angle: float):
    """(image, dc_mask) at the native grid, via the scoring loader."""
    try:
        from raw_camera_processor import load_camera_capture_for_citl
    except ImportError:
        from citl_capture import load_camera_capture_for_citl
    I_cam, _, dc_mask, _ = load_camera_capture_for_citl(
        image_path=str(path), roi=rig_roi, out_hw=NATIVE_HW, device="cpu",
        angle=angle, dc_radius=DC_RADIUS_NATIVE, auto_center=False,
        dc_center=DC_CENTER_NATIVE, subtract_min=True)
    return (I_cam[0].numpy().astype(np.float32),
            dc_mask.numpy().astype(bool))


def _load_target(target: str) -> np.ndarray:
    stock = Path(os.environ["SAILREV_RESULTS"]) / "Stock Images" / "1000x1000"
    for ext in (".png", ".jpg", ".jpeg"):
        p = stock / f"{target}{ext}"
        if p.exists():
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            return g / (np.percentile(g, TARGET_PERCENTILE) + 1e-9)
    return np.zeros(NATIVE_HW, dtype=np.float32)


def _roi_pixels(target: str, shape, rois: dict):
    """(y0, x0, side, manual?) -- GD+CITL convention: fractions are
    (y0, x0, side) of the frame; unlisted targets get the FULL frame so nothing
    is ever silently zoomed into another target's region."""
    H, W = shape
    if target not in rois:
        side = min(H, W)
        return (H - side) // 2, (W - side) // 2, side, False
    y0f, x0f, sf = rois[target]
    side = int(round(sf * min(H, W)))
    y0 = max(0, min(int(round(y0f * H)), H - side))
    x0 = max(0, min(int(round(x0f * W)), W - side))
    return y0, x0, side, True


def build(out_dir, columns=None):
    """E8, two products per (physics, target), Dilawer 2026-08-04:

    1. e8_grid/  : target at left spanning a 2x3 grid of the six methods,
                   full frames, PSNR badge per capture.
    2. e8_roi/   : the GD+CITL-style strip: full target with the green ROI box,
                   then the zoomed ROI of the target and of every method, PSNR
                   in each title. ROIs from roi.json ((y0, x0, side) fractions,
                   GD+CITL convention); unlisted targets render full-frame.

    PSNR values are the CANONICAL full-frame scored values (never recomputed
    on the crop), same principle as the GD+CITL figures. Method panels share
    one display scale per target (joint 99.7th percentile); the target image
    gets its own. FNO excluded: purely experimental figure.
    """
    import json as _json
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    S.apply_style()
    columns = list(columns or COLUMNS)

    run = _capture_run()
    roi, angle = _calibration(run)
    roi_path = Path(__file__).parent / ROI_FILENAME
    rois = _json.loads(roi_path.read_text()) if roi_path.exists() else {}
    print(f"E8 | captures from {run.name}; ROIs for {len(rois)} targets "
          f"from {roi_path.name}")

    grids_root = Path(out_dir) / "e8_grid"
    roi_root = Path(out_dir) / "e8_roi"
    for physics in S.PHYSICS:
        psnr = {m: S.by_target("bench", physics, m) for m in columns}
        ts = sorted(set.union(*[set(v) for v in psnr.values()]))
        for t in ts:
            # native-grid images through the scoring loader, once per target
            crops = {}
            for m in columns:
                f = run / physics / t / f"{FILE_FOR.get(m, m)}.jpg"
                if f.exists():
                    crops[m] = _load_cam(f, roi, angle)
            if not crops:
                continue
            # joint display scale over the methods, DC excluded, so the zero
            # order clips to white instead of setting the scale
            vmax = max(float(np.percentile(img[~dc], CAPTURE_PERCENTILE))
                       for img, dc in crops.values())
            tgt = np.clip(_load_target(t), 0, 1)

            def tile(m):
                return np.clip(crops[m][0] / (vmax + 1e-9), 0, 1)

            # ---- 1. the 2x3 grid --------------------------------------
            fig = plt.figure(figsize=(13.6, 6.6))
            gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1],
                                  wspace=0.08, hspace=0.08)
            ax = fig.add_subplot(gs[:, 0])
            ax.imshow(tgt, cmap="gray", vmin=0, vmax=1)
            ax.set_title("Target", fontsize=S.SIZE["title"], pad=6)
            ax.axis("off")
            for k, m in enumerate(columns):
                ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
                if m in crops:
                    ax.imshow(tile(m), cmap="gray", vmin=0, vmax=1,
                              interpolation="antialiased")
                ax.set_title(SHORT.get(m, S.label(m)),
                             fontsize=S.SIZE["title"], pad=6)
                v = psnr[m].get(t)
                if v is not None:
                    ax.text(0.03, 0.03, f"{v:.2f} dB", color="white",
                            fontsize=S.SIZE["annot"], fontweight="bold",
                            ha="left", va="bottom", transform=ax.transAxes,
                            bbox=dict(facecolor="black", alpha=0.55,
                                      pad=1.6, edgecolor="none"))
                ax.axis("off")
            S.save(fig, grids_root / f"e8_{physics}_{t}", dpi=320)
            plt.close(fig)

            # ---- 2. the ROI strip -------------------------------------
            shape = next(iter(crops.values()))[0].shape
            y0, x0, side, manual = _roi_pixels(t, shape, rois)
            # target ROI in target-image coordinates (same fractions)
            ty0, tx0, tside, _ = _roi_pixels(t, tgt.shape, rois)
            ncol = 2 + len(columns)
            # 2.5 in per panel at dpi 400 = 1000 px: native resolution
            # survives into the saved file.
            fig, axes = plt.subplots(1, ncol,
                                     figsize=(2.5 * ncol, 3.3),
                                     squeeze=False)
            axes = axes[0]
            axes[0].imshow(tgt, cmap="gray", vmin=0, vmax=1,
                           interpolation="antialiased")
            if manual:
                axes[0].add_patch(Rectangle((tx0, ty0), tside, tside,
                                            fill=False,
                                            edgecolor=ROI_BOX_COLOUR,
                                            linewidth=1.4))
            axes[0].set_title("Target", fontsize=S.SIZE["tick"], pad=5)
            axes[0].axis("off")
            # "(ROI)" removed from panel titles (caption carries it); the
            # green box and borders mark the zoom.
            suffix = ""
            axes[1].imshow(tgt[ty0:ty0 + tside, tx0:tx0 + tside],
                           cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest")
            # The zoomed target keeps "(ROI)" so the two target panels are
            # distinguishable; method panels stay unsuffixed (caption covers
            # it) -- Dilawer 2026-08-04.
            axes[1].set_title("Target (ROI)" if manual else "Target",
                              fontsize=S.SIZE["tick"], pad=5)
            for j, m in enumerate(columns, start=2):
                ax = axes[j]
                if m in crops:
                    z = crops[m][0][y0:y0 + side, x0:x0 + side]
                    ax.imshow(np.clip(z / (vmax + 1e-9), 0, 1), cmap="gray",
                              vmin=0, vmax=1, interpolation="nearest")
                v = psnr[m].get(t)
                label = SHORT.get(m, S.label(m))
                ax.set_title(f"{label}{suffix}" +
                             (f"\nPSNR {v:.2f} dB" if v is not None else ""),
                             fontsize=S.SIZE["tick"], pad=5, linespacing=1.3)
            for ax in axes[1:]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(ROI_BOX_COLOUR); sp.set_linewidth(1.2)
            fig.subplots_adjust(wspace=0.04)
            S.save(fig, roi_root / f"e8_roi_{physics}_{t}", dpi=400)
            plt.close(fig)
    return None
