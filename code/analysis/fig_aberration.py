r"""
fig_aberration.py -- E9 (aberration severity sweep) and Fig 5 (wavefront
control under misalignment, SAIL alone).

Both read scored_aberration.json, written by aberration_analysis.score();
nothing is recomputed here, so the number on a panel is by construction
the number in the scored file.

E9, two panels. Reconstruction quality against lens displacement, three
arms: the simulation hologram (never adapted to any rig), the SAIL
hologram adapted to the aligned bench, and SAIL retrained at each
position. Panel (a) is PSNR and panel (b) SSIM, both plotted because on
this target they carry very different amounts of information: the PSNR
spread across the whole sweep is under half a decibel, while SSIM moves
by more than a factor of two. Panel (b)'s inset reports the converse
control, the 2 mm-trained hologram photographed back at zero
displacement.

X axis: symmetric-log with a linear region below 0.05 mm, so the zero
point sits on the axis. Displacements are nominal caliper settings,
roughly +/-0.05 mm; the 0.05 mm point is at the edge of what could be
set repeatably.

Fig 5. One chosen displacement rendered through the same ROI machinery as
E8 (roi.json, native-resolution crops, canonical PSNR in the titles),
showing target, the never-adapted hologram, the saved SAIL hologram and
the retrained one. SAIL alone; no corrector anywhere in this figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sailrev as S
import aberration_analysis as A

LINTHRESH = 0.05          # mm; linear below this so zero sits on the axis
ARM_STYLE = {
    # arm: (colour, marker, linestyle, legend label)
    "transformer": (S.PURPLE_F[0], "o", "--",
                    "Simulation hologram (never adapted)"),
    "sail_saved": (S.BLUE[1], "s", "-",
                   "SAIL adapted to the aligned bench"),
    "sail_retrained": (S.BLUE[2], "D", "-",
                       "SAIL retrained at each position"),
}
PANELS = (("psnr", "PSNR (dB)", 1), ("ssim", "SSIM", 2))
# Fig 5 columns, worst to best left to right, matching E8's reading order.
FIG5_ARMS = ("transformer", "sail_saved", "sail_retrained")
FIG5_SHORT = {"transformer": "Simulation hologram",
              "sail_saved": "SAIL (aligned bench)",
              "sail_retrained": "SAIL (retrained)"}
ROI_BOX_COLOUR = "#2ee06a"


def _series(d, arm, metric):
    """[(mm, value)] for one arm, checks excluded, sorted by severity."""
    out = [(r["severity_mm"], r[metric]) for r in d["records"]
           if r.get("arm") == arm and not r.get("check")]
    return sorted(out)


def _checks(d):
    return {r["arm"]: r for r in d["records"] if r.get("check")}


# --------------------------------------------------------------------------
# E9
# --------------------------------------------------------------------------
def build(out_dir, path=None):
    import matplotlib.pyplot as plt
    S.apply_style()
    d = A.load(path)
    checks = _checks(d)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for ax, (metric, ylabel, _) in zip(axes, PANELS):
        for arm, (colour, marker, ls, _lab) in ARM_STYLE.items():
            xy = _series(d, arm, metric)
            if not xy:
                continue
            xs = [x for x, _ in xy]
            ys = [y for _, y in xy]
            ax.plot(xs, ys, color=colour, lw=2.6, ls=ls, marker=marker,
                    ms=7, markeredgecolor="white", markeredgewidth=1.0,
                    zorder=3)
        ax.set_xscale("symlog", linthresh=LINTHRESH,
                      linscale=0.7, subs=[2, 3, 4, 5, 6, 7, 8, 9])
        mms = sorted({x for a in ARM_STYLE
                      for x, _ in _series(d, a, metric)})
        ax.set_xticks(mms)
        ax.set_xticklabels([f"{m:g}" for m in mms])
        ax.set_xlabel("lens displacement toward the camera (mm)",
                      fontsize=S.SIZE["label"])
        ax.set_ylabel(ylabel, fontsize=S.SIZE["label"])
        S.style_axis(ax)

    axes[0].set_title("Reconstruction quality against defocus",
                      fontsize=S.SIZE["title"], pad=10)
    axes[1].set_title("Structural similarity, the sensitive metric here",
                      fontsize=S.SIZE["title"], pad=10)
    S.add_panel_label(axes[0], "a")
    S.add_panel_label(axes[1], "b")

    # The converse control, stated on the figure because it is the reason the
    # recovery arm cannot be read as "just a better hologram".
    ctrl = checks.get("control_2mm_at_zero")
    base_saved = _series(d, "sail_saved", "psnr")
    if ctrl and base_saved and base_saved[0][0] == 0.0:
        delta = ctrl["psnr"] - base_saved[0][1]
        axes[1].text(
            0.985, 0.96,
            "Control: the 2 mm-trained hologram\n"
            f"shown back at zero scores {delta:+.2f} dB\n"
            "against the zero-trained one",
            transform=axes[1].transAxes, ha="right", va="top",
            fontsize=S.SIZE["annot"], color="#212121",
            bbox=dict(facecolor="white", edgecolor="#e0e0e0",
                      boxstyle="round,pad=0.45", alpha=0.95))

    handles = [plt.Line2D([], [], color=c, lw=2.6, ls=ls, marker=mk, ms=7,
                          markeredgecolor="white", label=lab)
               for c, mk, ls, lab in ARM_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=S.SIZE["legend"], bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.0)
    S.save(fig, Path(out_dir) / "e9_aberration")
    plt.close(fig)

    _print_summary(d, checks)
    return None


def _print_summary(d, checks):
    saved = dict(_series(d, "sail_saved", "ssim"))
    sim = dict(_series(d, "transformer", "ssim"))
    fresh = dict(_series(d, "sail_retrained", "ssim"))
    if saved and sim:
        mm = sorted(saved)
        print(f"E9 | value of adaptation (SSIM above the never-adapted arm): "
              + ", ".join(f"{m:g} mm {saved[m] - sim[m]:+.3f}" for m in mm))
    if fresh:
        mm = sorted(set(fresh) & set(saved))
        print("E9 | recovery (retrained minus saved, SSIM): "
              + ", ".join(f"{m:g} mm {fresh[m] - saved[m]:+.3f}" for m in mm))
    else:
        print("E9 | NOTE: no retrained arm in the scored file; the recovery "
              "line is missing.\n     Re-run aberration_analysis.score() with "
              "the sail_runs tree present.")
    for k, what in (("return_sail_saved", "SAIL adapted to the aligned bench"),
                    ("return_transformer", "simulation hologram")):
        if k in checks:
            print(f"E9 | return-to-zero, {what}: {checks[k]['psnr']:.2f} dB")


# --------------------------------------------------------------------------
# Fig 5
# --------------------------------------------------------------------------
def build_fig5(out_dir, condition=None, path=None):
    """Fig 5: one displacement, target plus the three arms, ROI strip.

    condition defaults to the largest displacement present, which is where
    the three arms separate most clearly. Reuses fig_grid's loader and ROI
    convention so Fig 5 and E8 are rendered by identical machinery.
    """
    import json as _json
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    import fig_grid as G
    S.apply_style()

    d = A.load(path)
    rows = [r for r in d["records"] if not r.get("check")]
    if condition is None:
        condition = max(rows, key=lambda r: r["severity_mm"])["condition"]
    sel = {r["arm"]: r for r in rows if r["condition"] == condition}
    mm = next(iter(sel.values()))["severity_mm"] if sel else None
    if not sel:
        print(f"Fig 5 | nothing scored for condition {condition}")
        return None

    cal = d["meta"]["calibration"]
    roi, angle = cal["roi"], cal["rotation_deg"]
    roi_path = Path(G.__file__).parent / G.ROI_FILENAME
    rois = _json.loads(roi_path.read_text()) if roi_path.exists() else {}

    crops = {}
    for arm, r in sel.items():
        p = Path(r["source_path"])
        if p.exists():
            crops[arm] = G._load_cam(p, roi, angle)
    if not crops:
        print("Fig 5 | no captures readable")
        return None
    vmax = max(float(np.percentile(img[~dc], G.CAPTURE_PERCENTILE))
               for img, dc in crops.values())
    tgt = np.clip(G._load_target(A.TARGET), 0, 1)

    shape = next(iter(crops.values()))[0].shape
    y0, x0, side, manual = G._roi_pixels(A.TARGET, shape, rois)
    ty0, tx0, tside, _ = G._roi_pixels(A.TARGET, tgt.shape, rois)
    arms = [a for a in FIG5_ARMS if a in crops]
    ncol = 2 + len(arms)

    fig, axes = plt.subplots(1, ncol, figsize=(2.5 * ncol, 3.4), squeeze=False)
    axes = axes[0]
    axes[0].imshow(tgt, cmap="gray", vmin=0, vmax=1,
                   interpolation="antialiased")
    if manual:
        axes[0].add_patch(Rectangle((tx0, ty0), tside, tside, fill=False,
                                    edgecolor=ROI_BOX_COLOUR, linewidth=1.4))
    axes[0].set_title("Target", fontsize=S.SIZE["tick"], pad=5)
    axes[0].axis("off")
    axes[1].imshow(tgt[ty0:ty0 + tside, tx0:tx0 + tside], cmap="gray",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Target (ROI)" if manual else "Target",
                      fontsize=S.SIZE["tick"], pad=5)
    for j, arm in enumerate(arms, start=2):
        ax = axes[j]
        z = crops[arm][0][y0:y0 + side, x0:x0 + side]
        ax.imshow(np.clip(z / (vmax + 1e-9), 0, 1), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{FIG5_SHORT[arm]}\nPSNR {sel[arm]['psnr']:.2f} dB",
                     fontsize=S.SIZE["tick"], pad=5, linespacing=1.3)
    for ax in axes[1:]:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(ROI_BOX_COLOUR); sp.set_linewidth(1.2)
    fig.subplots_adjust(wspace=0.04)
    S.save(fig, Path(out_dir) / f"fig5_misalignment_{condition}", dpi=400)
    plt.close(fig)
    print(f"Fig 5 | {condition} ({mm:g} mm) built")
    return None
