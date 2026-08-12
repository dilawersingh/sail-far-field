r"""
fig_target_formulation.py -- E4. Amplitude versus intensity targets.

GS and GD run under both target formulations on all 18 targets under both
forward models. GS substitutes the chosen array as the replay-plane
magnitude constraint at every iteration, so it is directly sensitive to
the numeric scale of what it is handed; intensity = amplitude^2
compresses contrast, so intensity substitution hands GS a quadratically
distorted brightness structure. GD's loss is computed against recovered
intensity regardless of which array the formulation started from, so it
has no mismatch to absorb. The measured effect is accordingly large for
GS and negligible in magnitude for GD, under both forward models, and the
amplitude formulation is retained as the stronger configuration for the
baselines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sailrev as S

PAIRS = [("gs", "gs_intensity"), ("gd", "gd_intensity")]


def compute(path=None) -> dict:
    out = {}
    for physics in S.PHYSICS:
        rows = []
        for amp, inten in PAIRS:
            a = S.by_target("simulation", physics, amp, "psnr", path)
            i = S.by_target("simulation", physics, inten, "psnr", path)
            ts = sorted(set(a) & set(i))
            deltas = np.array([a[t] - i[t] for t in ts])
            d = S.summarize(deltas)
            d.update({"method": amp, "amp": a, "int": i, "targets": ts,
                      "deltas": deltas,
                      "amp_summary": S.summarize([a[t] for t in ts]),
                      "int_summary": S.summarize([i[t] for t in ts]),
                      "wins": int((deltas > 0).sum()),
                      "p": S.wilcoxon(deltas)[1] if np.any(deltas != 0) else np.nan})
            rows.append(d)
        out[physics] = rows
    return out


def report(d: dict) -> None:
    print("\nE4 | amplitude vs intensity targets (simulation, PSNR dB)")
    for physics, rows in d.items():
        print(f"  {S.PHYSICS_LABEL[physics]}")
        for r in rows:
            a, i = r["amp_summary"], r["int_summary"]
            print(f"    {S.label(r['method']):6s} amplitude {a['mean']:6.2f} "
                  f"+/- {a['sd']:5.2f}   intensity {i['mean']:6.2f} +/- "
                  f"{i['sd']:5.2f}   delta {r['median']:+6.2f} "
                  f"(amplitude better on {r['wins']}/{len(r['targets'])}, "
                  f"p={r['p']:.2e})")
    print("\n  GS: amplitude decisively stronger. GD: the effect is negligible in "
          "MAGNITUDE\n  (+0.16 dB ideal, -0.11 faithful) though the ideal GD "
          "difference is nominally\n  significant (p=0.027); do not write "
          "'statistically indistinguishable'.\n  Amplitude retained: the "
          "baselines are reported in their stronger configuration.")


def draw(d: dict, out_path):
    import matplotlib.pyplot as plt
    S.apply_style()

    # NOT sharey. Ideal and faithful are separate scoring problems, and the
    # ideal GS effect (+21 dB) is two orders larger than anything in the
    # faithful panel: a shared axis flattens that panel into a line and hides
    # the result it is there to show.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=False)
    rng = np.random.default_rng(0)

    for ax, physics, panel in zip(axes, S.PHYSICS, "ab"):
        rows = d[physics]
        labels = []
        ax.axhline(0, color=S.COLOR_DIAG, lw=1.0, ls="--", zorder=1)
        for i, r in enumerate(rows):
            c = S.method_color(r["method"])
            x = i + rng.uniform(-0.11, 0.11, r["deltas"].size)
            ax.scatter(x, r["deltas"], s=30, color=c, alpha=0.45,
                       edgecolors="none", zorder=2)
            ax.plot([i - 0.24, i + 0.24], [r["median"]] * 2, color=c, lw=3.0,
                    solid_capstyle="round", zorder=3)
            labels.append((i, r, c))
        # Labels are placed after the data so the axis limits are known: each
        # sits a fixed FRACTION of the panel's own range from its marker, which
        # keeps it clear of both the points and the axis whatever the scale.
        span = max(abs(v) for r in rows for v in r["deltas"]) or 1.0
        ax.set_ylim(min(0, min(v for r in rows for v in r["deltas"])) - 0.30 * span,
                    max(0, max(v for r in rows for v in r["deltas"])) + 0.22 * span)
        # Beside the median bar, not above it: the swarm occupies the vertical
        # space directly over each marker. The halo covers the residual case
        # where a stray point sits under the text.
        for i, r, c in labels:
            ax.annotate(f"{r['median']:+.2f} dB\n{r['wins']}/{len(r['targets'])}",
                        xy=(i + 0.26, r["median"]), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=S.SIZE["annot"], fontweight="bold", color=c,
                        bbox=S.halo())
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([S.label(r["method"]) for r in rows],
                           fontsize=S.SIZE["tick"])
        ax.set_xlim(-0.5, len(rows) - 0.05)
        ax.set_title(S.PHYSICS_LABEL[physics], fontsize=S.SIZE["title"], pad=10)
        ax.set_ylabel("Amplitude over intensity, \u0394 PSNR (dB)",
                      fontsize=S.SIZE["label"])
        S.style_axis(ax); S.add_panel_label(ax, panel)

    fig.tight_layout(w_pad=2.4)
    S.save(fig, out_path)
    return fig


def save_tables(d: dict, table_dir) -> None:
    """Persist the printed summary. Carries the per-arm medians as well as
    the paired-delta statistics."""
    header = ["physics", "method",
              "amp_median", "amp_mean", "amp_sd",
              "int_median", "int_mean", "int_sd",
              "delta_median", "delta_mean", "delta_sd",
              "delta_ci_lo", "delta_ci_hi", "wins", "n", "p"]
    rows = []
    for physics, prs in d.items():
        for r in prs:
            a, i = r["amp_summary"], r["int_summary"]
            rows.append([physics, S.label(r["method"]),
                         a["median"], a["mean"], a["sd"],
                         i["median"], i["mean"], i["sd"],
                         r["median"], r["mean"], r["sd"],
                         r["ci_lo"], r["ci_hi"],
                         r["wins"], len(r["targets"]), r["p"]])
    S.write_table(
        table_dir, "e4_target_formulation_summary", header, rows,
        note="E4 | amplitude vs intensity targets, simulation, PSNR (dB); "
             "delta = amplitude minus intensity, paired by target, Wilcoxon "
             "signed-rank p. Source: sail_scored.json (simulation domain = "
             "the matched-compute 10k sweep). The narrative quotes the "
             "median; take significance verdicts from the p column of this "
             "table, not from prose.")


def build(out_dir, path=None):
    d = compute(path)
    report(d)
    save_tables(d, Path(out_dir).parent / "tables")
    fig = draw(d, Path(out_dir) / "e4_amplitude_vs_intensity")
    return fig
