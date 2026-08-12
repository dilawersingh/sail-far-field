r"""
fig_generalisation.py -- E6. Generalisation to unseen patterns at the
synthetic proof-of-concept resolution.

WHY 28x28. This model is the proof-of-concept for CURRICULUM training:
    it was trained on a synthetic curriculum (Fourier basis functions,
    random noise, doodles) and then asked to produce phase holograms for
    intensities it had never seen. Full resolution was not a modelling
    limitation but a data-scale one: at 1080x1920 each sample is ~2 MB, and
    a curriculum spanning the full frequency set is ~2M samples, which is
    ~4 TB of training data -- beyond the single-workstation compute budget
    of this study. The 28x28 study establishes the capability; the
    full-resolution models in the rest of the paper are per-target or
    18-target batched precisely because of that budget.

WHAT IT SHOWS. The official clean test run, "Testing Synthetic Model on
Custom Intensity_20260218_214556" (Dilawer 2026-08-04): a 3,306,120-parameter
transformer (Dense 196>256, 2D sinusoidal PE, 4-layer encoder d_model=256,
16 heads, Dense 256>392) generating 28x28 phase holograms in ONE FORWARD PASS
for 16 patterns spanning families it was never trained on: Gaussian spots,
rings, squares, an MNIST digit, a Fashion-MNIST garment, hand doodles and
text. Rows: target far-field, predicted far-field (through the simulated
propagation), predicted phase. Columns keep the archive's target indices.

DATA PROVENANCE AND THE RECOVERY STEP. The 2026-02-18 run archived rendered
figures (target N.png: phase | predicted | target, identical layout), not
arrays. Rather than re-running a 2026 training script, this module RECOVERS
the underlying 28x28 arrays from the archived renders: the three panels are
located by connected-component analysis and each 28x28 grid is re-sampled at
cell centres (the renders are nearest-neighbour upsamplings, so cell-centre
sampling is exact up to 8-bit quantisation). The figure therefore shows the
run's own archived pixels, restyled; nothing is regenerated or retrained.

NO METRICS, DELIBERATELY. The archived renders are display-scaled 8-bit
images; PSNR computed from them would not be canonical and nothing
non-canonical gets quoted (project rule). E6 is the qualitative
generalisation demonstration; quantitative claims live in the scored
domains.

Reads {SAILREV_RESULTS}/Transformers/hologram_generation/synthetic/
multi-level/testing/Testing Synthetic Model on Custom Intensity_20260218_
214556/target *.png.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import sailrev as S

RUN_NAME = "Testing Synthetic Model on Custom Intensity_20260218_214556"
N = 28                       # native grid of the synthetic study
MIN_PANEL_AREA = 50_000      # px; the three imshow panels are ~523x522
ROW_LABELS = ["Target\nfar-field", "Predicted\nfar-field", "Predicted\nphase"]


def _run_dir() -> Path:
    return (Path(os.environ["SAILREV_RESULTS"]) / "Transformers" /
            "hologram_generation" / "synthetic" / "multi-level" / "testing" /
            RUN_NAME)


def _recover(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target, predicted, phase), each 28x28 in [0, 1], from one archived
    render. Panels are found as the three large square-ish connected
    components (robust to the two archive layouts: with and without titles);
    left-to-right in the archive they are phase | predicted | target."""
    img = np.asarray(Image.open(path).convert("L"))
    mask = (img < 250).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = sorted(
        [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)
         if stats[i][4] > MIN_PANEL_AREA
         and 0.8 < stats[i][2] / stats[i][3] < 1.25],
        key=lambda b: b[0])
    if len(boxes) != 3:
        raise ValueError(f"{path.name}: found {len(boxes)} panels, need 3")
    grids = []
    for x, y, w, h in boxes:
        ys = (y + (np.arange(N) + 0.5) * h / N).astype(int)
        xs = (x + (np.arange(N) + 0.5) * w / N).astype(int)
        grids.append(img[np.ix_(ys, xs)].astype(np.float32) / 255.0)
    phase, pred, tgt = grids
    return tgt, pred, phase


def build(out_dir):
    import matplotlib.pyplot as plt
    S.apply_style()

    run = _run_dir()
    files = sorted(run.glob("target *.png"),
                   key=lambda p: int(re.search(r"target (\d+)", p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"no 'target N.png' renders under {run}")
    print(f"E6 | {len(files)} unseen patterns from {run.name}")

    cols = [_recover(p) for p in files]
    nc = len(cols)
    fig, axes = plt.subplots(3, nc, figsize=(0.86 * nc + 1.7, 3 * 0.86 + 0.5),
                             squeeze=False)
    for c, (tgt, pred, phase) in enumerate(cols):
        for r, im in enumerate((tgt, pred, phase)):
            ax = axes[r][c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=S.SIZE["tick_small"],
                              rotation=0, ha="right", va="center", labelpad=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.075, right=0.995,
                        top=0.99, bottom=0.01)
    S.save(fig, Path(out_dir) / "e6_generalisation")
    return fig
