r"""
fig_patch.py -- E2. Patch-size sweep.

Patch size varied over 500, 250, 125 and 50 with embedding width, head
count, depth and training schedule held fixed. Median PSNR falls
monotonically on 18 of 18 targets as the patch shrinks. Shrinking the
patch widens the token bottleneck (4 tokens x 256 dims at p = 500 becomes
400 x 256 at p = 50) but shrinks the shared output head,
Linear(d_model, 2*p*p), from 128M parameters at p = 500 to 1.28M at
p = 50, and that single map is what produces the phase. Far-field phase
is speckle and essentially incompressible, so the head, not the token
count, is the binding constraint at this configuration.

Reads patch_sweep_summary.json rather than sail_scored.json: the sweep is
a 72-run architecture study (4 patch sizes x 18 targets), scored by the
same compute_metrics inside the sweep itself, and it is not part of the
canonical method comparison.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import sailrev as S

PATCHES = [500, 250, 125, 50]
TOKENS = {500: 4, 250: 16, 125: 64, 50: 400}


def summary_path(path=None) -> Path:
    if path:
        return Path(path)
    if os.environ.get("SAILREV_PATCH_SWEEP"):
        return Path(os.environ["SAILREV_PATCH_SWEEP"])
    root = os.environ.get("SAILREV_SA") or os.path.join(
        os.environ["SAILREV_RESULTS"], "Self-Attention", "multilevel")
    return Path(root) / "simulations" / "patch_sweep" / "patch_sweep_summary.json"


def compute(path=None, physics: str = "ideal") -> dict:
    rows = [r for r in json.loads(Path(summary_path(path)).read_text())
            if r["physics_name"] == physics]
    by_patch = defaultdict(dict)
    params = {}
    for r in rows:
        by_patch[int(r["patch"])][r["image"]] = float(r["psnr"])
        params[int(r["patch"])] = int(r["params"])
    summary = {p: S.summarize(list(v.values())) for p, v in by_patch.items()}
    # Did every target degrade, or only the mean? A mean can fall while some
    # targets improve; "monotonic on 18 of 18" is a much stronger statement and
    # is checked here rather than asserted.
    targets = sorted(by_patch[PATCHES[0]])
    monotonic = sum(
        1 for t in targets
        if all(by_patch[a][t] >= by_patch[b][t]
               for a, b in zip(PATCHES, PATCHES[1:])))
    return {"by_patch": dict(by_patch), "summary": summary, "params": params,
            "targets": targets, "monotonic": monotonic, "physics": physics}


def report(d: dict) -> None:
    print(f"\nE2 | patch sweep ({S.PHYSICS_LABEL[d['physics']]}, "
          f"{len(d['targets'])} targets)")
    print(f"  {'patch':>6s} {'tokens':>7s} {'params':>14s} "
          f"{'mean PSNR':>10s} {'median':>8s}")
    for p in PATCHES:
        s = d["summary"][p]
        print(f"  {p:6d} {TOKENS[p]:7d} {d['params'][p]:14,d} "
              f"{s['mean']:10.2f} {s['median']:8.2f}")
    print(f"\n  degradation is monotonic on {d['monotonic']}/"
          f"{len(d['targets'])} targets")
    best, worst = d["summary"][500]["mean"], d["summary"][50]["mean"]
    print(f"  p=500 over p=50: {best - worst:.2f} dB")


def draw(d: dict, out_path):
    import matplotlib.pyplot as plt
    S.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    xs = np.arange(len(PATCHES))
    # Sequential quantity (one architecture, four settings), so one hue stepped
    # by lightness rather than four categorical colours.
    ramp = [S.method_color("sail"), "#4a9fe0", "#7fbde8", "#b9d9f2"]

    # ---- (a) per-target traces + mean ------------------------------------
    ax = axes[0]
    for t in d["targets"]:
        ax.plot(xs, [d["by_patch"][p][t] for p in PATCHES],
                color=S.COLOR_FAINT, lw=1.0, alpha=0.7, zorder=1)
    means = [d["summary"][p]["mean"] for p in PATCHES]
    # Same ramp as panel (b) (Dilawer 2026-08-04): the line segments stay in
    # the SAIL blue, but each marker takes its patch size's ramp colour so the
    # two panels read as one gradient.
    ax.plot(xs, means, color=S.method_color("sail"), lw=3.0, zorder=3,
            label="mean over 18 targets")
    for x, m, rc in zip(xs, means, ramp):
        ax.plot([x], [m], marker="o", ms=12, color=rc, linestyle="none",
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
    # The mean line descends steeply, so a label centred above a marker lands
    # on the segment leaving it. Offsetting to the upper LEFT puts each label
    # in the wedge the line has already passed through, and the halo keeps it
    # readable over the faint per-target traces behind.
    for x, m in zip(xs, means):
        # First point sits on the y-axis, so its label flips to the right.
        dx, align = ((10, "left") if x == xs[0] else (-10, "right"))
        ax.annotate(f"{m:.2f}", xy=(x, m), xytext=(dx, 12),
                    textcoords="offset points", ha=align, va="bottom",
                    fontsize=S.SIZE["annot"], fontweight="bold",
                    color=S.method_color("sail"), bbox=S.halo())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p = {p}\n{TOKENS[p]} tokens" for p in PATCHES],
                       fontsize=S.SIZE["tick_small"])
    ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
    ax.set_xlim(-0.35, len(PATCHES) - 0.65)
    ax.set_title("Finer patches are monotonically worse\n"
                 f"({d['monotonic']} of {len(d['targets'])} targets)",
                 fontsize=S.SIZE["title"], pad=10)
    ax.legend(frameon=False, fontsize=S.SIZE["legend"], loc="lower left")
    S.style_axis(ax); S.add_panel_label(ax, "a")

    # ---- (b) the head is the binding constraint --------------------------
    ax = axes[1]
    head = [2 * p * p * 256 for p in PATCHES]      # Linear(d_model, 2*p*p)
    ax.bar(xs, head, width=0.6, color=ramp, edgecolor="white", linewidth=1.2)
    for x, h in zip(xs, head):
        ax.text(x, h * 1.25, f"{h/1e6:.1f}M", ha="center",
                fontsize=S.SIZE["annot"], fontweight="bold",
                color=S.COLOR_DIAG, bbox=S.halo())
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p = {p}" for p in PATCHES], fontsize=S.SIZE["tick"])
    ax.set_ylabel("output head parameters (log)", fontsize=S.SIZE["label"])
    ax.set_ylim(top=max(head) * 6)
    ax.set_title("The head shrinks 100-fold:\nthe binding constraint",
                 fontsize=S.SIZE["title"], pad=10)
    S.style_axis(ax); S.add_panel_label(ax, "b")

    fig.tight_layout(w_pad=2.6)
    S.save(fig, out_path)
    return fig


def save_tables(d: dict, table_dir) -> None:
    """Persist the printed summary."""
    S.write_table(
        table_dir, "e2_patch_sweep_summary",
        ["patch", "tokens", "params"] + S.SUMMARY_HEADER,
        [[p, TOKENS[p], d["params"][p]] + S.summary_row(d["summary"][p])
         for p in PATCHES],
        note=f"E2 | patch sweep, {S.PHYSICS_LABEL[d['physics']]}, PSNR (dB), "
             f"n={len(d['targets'])} targets; degradation monotonic on "
             f"{d['monotonic']}/{len(d['targets'])} targets. Source: "
             "patch_sweep_summary.json (72-run architecture study). "
             "The narrative quotes the median.")


def build(out_dir, path=None, physics: str = "ideal"):
    d = compute(path, physics)
    report(d)
    save_tables(d, Path(out_dir).parent / "tables")
    fig = draw(d, Path(out_dir) / "e2_patch_sweep")
    return fig
