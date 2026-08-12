r"""
fig_simulation.py -- Fig 2. The simulation benchmark at the operating point.

WHAT THIS FIGURE CLAIMS, PRECISELY. At the published operating point (750
iterations, itself roughly three times typical CGH practice) a single forward
pass matches or exceeds the iterative baselines per target, for roughly three
orders of magnitude less compute per hologram. It deliberately does NOT claim
superior converged quality: E5 shows GD overtaking the per-target transformer
in ideal simulation at ~5,000 iterations, and the manuscript concedes that
plainly (R2.2). This figure and E5 are a pair: Fig 2 is the operating point,
E5 is what unbounded compute does to it. The caption must say so and point to
E5, or a reviewer reads Fig 2 as the claim R2 already objected to.

PANELS
(a) Ideal model: per-target PSNR distributions (18 dots, median + IQR bar)
    for GS, GD, per-target transformer, batched transformer.
(b) Faithful model: same. Kept as a separate panel, never a shared axis:
    ideal and faithful are different scoring problems and the collapse of
    every method under the faithful model (49 -> 15 dB for the transformer)
    is itself a finding the reader should see, not a scale nuisance.
(c) Parity, per target: transformer vs GD at 750, both physics. Points above
    the diagonal are targets where one forward pass beats 750 GD iterations.
    Pairing is by target name.

Compute numbers quoted in the caption, not drawn: GS 0.6 s and GD 2.2 s per
target at 750 iterations (ideal; several-fold slower faithful), against a 13
to 23 ms forward pass (range reflects uncontrolled machine load; slow end
quoted wherever a single figure is needed). Sources: the sweep aggregates and
T2; this module reads sail_scored.json only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sailrev as S

METHODS = ["gs", "gd", "transformer_per_target", "transformer_batched"]


def _dist_panel(ax, physics):
    rng = np.random.default_rng(0)
    for i, m in enumerate(METHODS):
        v = np.array(list(S.by_target("simulation", physics, m).values()))
        if v.size == 0:
            continue
        c = S.method_color(m)
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=22, color=c,
                   alpha=0.55, linewidths=0, zorder=2)
        med, q1, q3 = np.median(v), *np.percentile(v, [25, 75])
        ax.hlines(med, i - 0.3, i + 0.3, color=c, lw=3.2, zorder=4)
        ax.vlines(i, q1, q3, color=c, lw=1.6, zorder=3)
        ax.text(i + 0.34, med, f"{med:.1f}", ha="left", va="center",
                fontsize=S.SIZE["annot"], color=c, fontweight="bold",
                bbox=S.halo())
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([S.label(m).replace(" ", "\n") for m in METHODS],
                       fontsize=S.SIZE["tick"])
    ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
    ax.set_title(f"{S.PHYSICS_LABEL[physics]}, {S.N_TARGETS} targets, "
                 f"iteration budget 750", fontsize=S.SIZE["title"], pad=10)
    S.style_axis(ax)


def build(out_dir):
    import matplotlib.pyplot as plt
    S.apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))

    _dist_panel(axes[0], "ideal")
    S.add_panel_label(axes[0], "a")
    _dist_panel(axes[1], "faithful")
    S.add_panel_label(axes[1], "b")

    # ---- (c) parity ------------------------------------------------------
    ax = axes[2]
    lims = [np.inf, -np.inf]
    for physics, marker, filled in (("ideal", "o", True),
                                    ("faithful", "s", False)):
        ts, d = S.paired("simulation", physics, "transformer_per_target", "gd")
        gd = S.by_target("simulation", physics, "gd")
        tr = S.by_target("simulation", physics, "transformer_per_target")
        x = np.array([gd[t] for t in ts]); y = np.array([tr[t] for t in ts])
        c = S.method_color("transformer_per_target")
        ax.scatter(x, y, s=42, marker=marker,
                   color=c if filled else "none",
                   edgecolors=c, linewidths=1.6, alpha=0.85,
                   label=f"{S.PHYSICS_LABEL[physics]} "
                         f"({int((d > 0).sum())}/{len(ts)} above)")
        lims = [min(lims[0], x.min(), y.min()), max(lims[1], x.max(), y.max())]
    pad = 0.05 * (lims[1] - lims[0])
    lo, hi = lims[0] - pad, lims[1] + pad
    ax.plot([lo, hi], [lo, hi], color=S.COLOR_DIAG, lw=1.2, ls="--", zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("GD, 750 iterations (dB)", fontsize=S.SIZE["label"])
    ax.set_ylabel("Transformer, one forward pass (dB)",
                  fontsize=S.SIZE["label"])
    ax.set_title("Per-target parity at the operating point",
                 fontsize=S.SIZE["title"], pad=10)
    ax.legend(frameon=False, fontsize=S.SIZE["legend"], loc="lower right")
    S.style_axis(ax)
    S.add_panel_label(ax, "c")

    fig.tight_layout(w_pad=3.0)
    S.save(fig, Path(out_dir) / "fig2_simulation")
    print("Fig 2 | caption must: state 750 is ~3x typical practice; quote the "
          "compute ratio\n(GS 0.6 s, GD 2.2 s per target vs a 13-23 ms forward "
          "pass, slow end quoted); and\npoint to E5 for the matched-compute "
          "sweep where GD overtakes in ideal simulation.")
    return fig
