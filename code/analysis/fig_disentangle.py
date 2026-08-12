r"""
fig_disentangle.py -- Fig 3. SAIL against every alternative on the bench.

THE QUESTION THIS FIGURE ANSWERS: is SAIL's advantage the architecture, or
is it the camera-in-the-loop adaptation that any method could have had? The
answer is one panel per forward model showing the paired
per-target difference between SAIL and EVERY other method measured on the
bench, including the classical baselines after they have been given the same
camera feedback.

Eight comparisons, same 18 targets, all differences SAIL minus the other:

    GS, GD                        classical, no adaptation
    Transformer                   our architecture, no adaptation
    Transformer (shared)          shared model, no adaptation
    GS+CITL (random, warm)        classical + camera feedback
    GD+CITL (random, warm)        classical + camera feedback, the strong one

Every row positive is the claim. The rows that matter most to a sceptical
reader are the two GD+CITL ones, because those hold adaptation constant and
vary only the architecture.

WHAT IS DELIBERATELY NOT PLOTTED HERE. The contrasts that quantify what
adaptation alone buys a classical method (GD+CITL - GD, GS+CITL - GS) are
computed and printed by adaptation_contrasts(), but they are not on this
figure: it exists to place SAIL
against the alternatives, and a reader does not need GD's internal improvement
to read that. The GS+CITL failure (best_iteration = 0 on all 18, camera
feedback leaves the RANDOM-INIT arm below plain GS; the simulation-seeded arm
sits marginally above it, +0.11 dB, which follows from best_iteration = 0
meaning it simply kept its seed) is still reported, in the text and in T1,
because a baseline of ours that breaks is worth stating plainly.

WHY BOTH FORWARD MODELS. Ideal and faithful are separate scoring problems
(different resolutions, so PSNR and SSIM are not comparable across them). They
are shown side by side and never merged; the claim is that the ORDERING is
stable under both, which is stronger than either alone.

Statistics: median of the paired differences with a seeded bootstrap interval,
paired Wilcoxon, per sailrev's fixed conventions. Means are in T1; the median
leads here because n=18 and the distributions are skewed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sailrev as S

# Everything SAIL is compared against. Row order is CANONICAL (structural),
# set in compute(); see the note there for why it is not sorted by median.
OTHERS = ["gs", "gd", "gs_citl_random", "gs_citl_warm",
          "gd_citl_random", "gd_citl_warm",
          "transformer_per_target", "transformer_batched"]

# Computed for adaptation_contrasts(), not drawn on this figure.
ADAPTATION = [("gd_citl_random", "gd"), ("gd_citl_warm", "gd"),
              ("gs_citl_random", "gs"), ("gs_citl_warm", "gs")]


def compute(path=None) -> dict:
    out = {}
    for physics in S.PHYSICS:
        rows = []
        for b in OTHERS:
            d = S.delta_summary("bench", physics, "sail", b, "psnr", path)
            _, deltas = S.paired("bench", physics, "sail", b, "psnr", path)
            d.update({"label": S.label(b), "deltas": deltas, "other": b})
            rows.append(d)
        out[physics] = rows

    # Rows follow the CANONICAL (structural) order: GS lineage, GD lineage,
    # then the learned arms. Two reasons, and neither is aesthetic alone.
    # First, both panels then place the same method on the same row, so a and b
    # are row-comparable. Second, the palette is assigned along this same
    # order, so the colours run smoothly down the y axis instead of jumping.
    # Sorting by measured median instead would let new data repaint the figure.
    order = [m for m in S.CANONICAL_ORDER if m in
             {r["other"] for r in out["ideal"]}][::-1]
    for physics in out:
        out[physics].sort(key=lambda r: order.index(r["other"]))
    return out


def report(data: dict) -> None:
    for physics, rows in data.items():
        print(f"\nSAIL minus each method ({S.PHYSICS_LABEL[physics]}, bench, "
              f"n=18, PSNR dB)")
        print(f"  {'vs':28s} {'median':>8s} {'95% CI':>18s} {'mean':>8s} "
              f"{'wins':>8s} {'p':>10s}")
        for r in reversed(rows):
            print(f"  {r['label']:28s} {r['median']:+8.2f} "
                  f"[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] {r['mean']:+8.2f} "
                  f"{r['wins']:3d}/{r['n_pairs']:<4d} {r['p']:10.2e}")


def adaptation_contrasts(path=None) -> None:
    """What camera feedback alone buys a classical method. For R1.5's text.

    Printed, never plotted here. The honest framing the skeleton insists on:
    adaptation is worth about as much as the architecture adds on top of it,
    so the defensible claim is that both matter and neither alone suffices.
    """
    print("\nAdaptation alone (not plotted; for the R1.5/R2.6 response)")
    for physics in S.PHYSICS:
        print(f"  {S.PHYSICS_LABEL[physics]}")
        for a, b in ADAPTATION:
            d = S.delta_summary("bench", physics, a, b, "psnr", path)
            print(f"    {S.label(a):24s} - {S.label(b):4s} "
                  f"median {d['median']:+6.2f} "
                  f"[{d['ci_lo']:+5.2f},{d['ci_hi']:+5.2f}] "
                  f"wins {d['wins']:2d}/{d['n_pairs']}")


def draw(data: dict, out_path, path=None):
    import matplotlib.pyplot as plt
    S.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), sharey=False)
    rng = np.random.default_rng(0)          # seeded jitter: identical rebuilds

    for ax, physics, panel in zip(axes, S.PHYSICS, "ab"):
        rows = data[physics]
        n = len(rows)

        # Reserve a clean annotation column on the right rather than writing
        # over the points. Text sat on top of the markers and the zero line in
        # the first version; a dedicated column cannot collide by construction.
        allv = np.concatenate([r["deltas"] for r in rows])
        lo, hi = float(allv.min()), float(allv.max())
        span = hi - lo or 1.0
        x_lo = min(0.0, lo) - 0.06 * span
        x_data_hi = hi + 0.04 * span
        x_text = x_data_hi + 0.06 * span        # left edge of the text column
        x_hi = x_text + 0.42 * span             # room for "+9.99 [..] 18/18"

        ax.axvline(0, color=S.COLOR_DIAG, lw=1.0, ls="--", zorder=1)
        for i, r in enumerate(rows):
            y = i
            d = r["deltas"]
            # One fixed colour per method, from sailrev's palette: violet for
            # methods that are not ours, pink for ours, darker within each hue
            # as more machinery is added. Never chosen per figure.
            c = S.method_color(r["other"])
            ax.scatter(d, y + rng.uniform(-0.15, 0.15, d.size), s=26,
                       color=c, alpha=0.42, edgecolors="none", zorder=2)
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2.6,
                    solid_capstyle="round", zorder=3)
            ax.plot([r["median"]], [y], marker="o", ms=9, color=c,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            ax.text(x_text, y, f"{r['median']:+.2f}  {r['wins']}/{r['n_pairs']}",
                    ha="left", va="center", fontsize=S.SIZE["tick_small"], color=c,
                    fontweight="bold")

        ax.set_yticks(range(n))
        ax.set_yticklabels([r["label"] for r in rows], fontsize=S.SIZE["tick"])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("Δ PSNR (dB), SAIL minus method", fontsize=S.SIZE["tick"])
        ax.set_title(S.PHYSICS_LABEL[physics], fontsize=S.SIZE["title"], pad=10)
        S.style_axis(ax)
        S.add_panel_label(ax, panel)

    fig.tight_layout(w_pad=3.2)
    S.save(fig, out_path)
    return fig


def build(out_dir, path=None):
    data = compute(path)
    report(data)
    adaptation_contrasts(path)
    fig = draw(data, Path(out_dir) / "fig3_sail_vs_all", path)
    print("\nReading: every row is positive under both forward models. The "
          "GD+CITL rows are the\nload-bearing ones, since they hold camera "
          "adaptation constant and vary only the\narchitecture. Numbers at "
          "the right are median Δ and targets won of 18.")
    return fig
