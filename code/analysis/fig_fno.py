r"""
fig_fno.py -- E1. The Fourier neural operator baseline.

A standard, unmodified FNO trained per target under the ideal forward
model, in two arms. Trained from scratch on the synthesis objective it
does not find a solution; trained by regression onto gradient descent's
converged phase it reproduces GD's quality, so the failure is
trainability rather than capacity. A learning-rate probe spanning five
orders of magnitude leaves the from-scratch arm unchanged.

Protocol: full mode coverage (modes = 500 on a 1000x1000 grid, the exact
ceiling, so nothing is spectrally truncated); 256.0M parameters against
the self-attention model's 195.7M, so FNO is given more capacity; the
same 10,000 epochs; coordinate channels enabled; identical forward model
and identical compute_metrics scoring as every other number in the
record. Ideal physics only: the from-scratch quality sits so far below
every other method that a faithful re-run cannot inform the comparison.

Panel (a) reads sail_scored.json. Panel (b) reads the learning-rate probe
summary, which is outside the canonical dataset because it is a
hyperparameter study rather than a scored reconstruction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sailrev as S

# All four arms are scored per target from the canonical dataset. The
# self-attention model is one of them, NOT a constant: an earlier version drew
# a flat line at 52.30 dB, which is alley's value alone, across a chart of 18
# targets. Comparing a per-target bar against a single-target reference
# overstates the model on every target where alley happens to be its best, so
# the line is gone and the model is scored the same way as everything else.
ARMS = ["fno_scratch", "fno_regress", "gd", "transformer_per_target"]


def lr_probe_path(path=None) -> Path:
    if path:
        return Path(path)
    if os.environ.get("SAILREV_LR_PROBE"):
        return Path(os.environ["SAILREV_LR_PROBE"])
    root = os.environ.get("SAILREV_FNO") or (
        os.path.join(os.environ["SAILREV_RESULTS"], "FNO"))
    return Path(root) / "lr_probe_w8m500" / "fno_sweep_summary.json"


def load_lr_probe(path=None) -> dict:
    """{lr: {target: psnr}} from the width=8, modes=500 probe."""
    rows = json.loads(Path(lr_probe_path(path)).read_text())
    out: dict[float, dict[str, float]] = {}
    for r in rows:
        out.setdefault(float(r["learning_rate"]), {})[r["target"]] = \
            float(r["best_psnr"])
    return out


def compute(path=None, lr_path=None) -> dict:
    per_target = {a: S.by_target("simulation", "ideal", a, "psnr", path)
                  for a in ARMS}
    summary = {a: S.summarize(list(v.values())) for a, v in per_target.items()}
    return {"per_target": per_target, "summary": summary,
            "lr": load_lr_probe(lr_path)}


def report(d: dict) -> None:
    print("\nE1 | FNO, simulation, ideal physics, 18 targets (PSNR dB)")
    for a in ARMS:
        s = d["summary"][a]
        print(f"  {S.label(a):32s} mean {s['mean']:6.2f} +/- {s['sd']:5.2f}   "
              f"median {s['median']:6.2f}")
    scratch = d["summary"]["fno_scratch"]["mean"]
    regress = d["summary"]["fno_regress"]["mean"]
    gdm = d["summary"]["gd"]["mean"]
    print(f"\n  regression reaches {regress:.2f} against GD's own {gdm:.2f}: "
          f"the architecture CAN represent the solution.")
    print(f"  training from scratch reaches {scratch:.2f}: it cannot FIND it.")
    print("\n  learning-rate probe (width 8, modes 500, 1000 epochs):")
    for lr in sorted(d["lr"], reverse=True):
        vals = list(d["lr"][lr].values())
        print(f"    lr {lr:<8g} mean PSNR {np.mean(vals):5.2f}  "
              f"({', '.join(f'{t} {v:.2f}' for t, v in d['lr'][lr].items())})")


def draw(d: dict, out_path):
    import matplotlib.pyplot as plt
    S.apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- (a) per-target: scratch vs regression vs GD ---------------------
    ax = axes[0]
    targets = sorted(d["per_target"]["gd"], key=lambda t: -d["per_target"]["gd"][t])
    x = np.arange(len(targets))
    w = 0.21
    for i, a in enumerate(ARMS):
        vals = [d["per_target"][a][t] for t in targets]
        ax.bar(x + (i - 1.5) * w, vals, width=w, color=S.method_color(a),
               edgecolor="white", linewidth=0.7, label=S.label(a))
    ax.set_ylim(0, max(v for a in ARMS
                       for v in d["per_target"][a].values()) * 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=60, ha="right",
                       fontsize=S.SIZE["tick_small"])
    ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
    ax.set_title("FNO reproduces GD's phase but cannot reach it from scratch",
                 fontsize=S.SIZE["title"], pad=64)
    # Legend ABOVE the axes. Inside the panel it sat on the mid-height bars,
    # and below the axes it sat on the rotated target names; there is no free
    # region inside a bar chart this dense, so it goes outside.
    # ncol=2 keeps the legend clear of the panel letter, which sits at the
    # top-left corner; a single four-column row ran straight through it.
    # Centered over the graph (Dilawer 2026-08-04); still above the axes so
    # it never touches the bars, ncol=2 still clears the panel letter.
    ax.legend(frameon=False, fontsize=S.SIZE["legend"], ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.4, handlelength=1.4)
    S.style_axis(ax); S.add_panel_label(ax, "a")

    # ---- (b) learning-rate insensitivity ---------------------------------
    ax = axes[1]
    lrs = sorted(d["lr"], reverse=True)
    all_targets = sorted({t for v in d["lr"].values() for t in v})
    for j, t in enumerate(all_targets):
        ys = [d["lr"][lr].get(t, np.nan) for lr in lrs]
        # Distinct from panel (a)'s legend colours, which mean CONDITIONS
        # there; here the lines are TARGETS and must not be read as conditions.
        # Widely stepped single-hue ramp: the lines are TARGETS, an ordered
        # sample, not conditions; COOL's adjacent steps were too similar
        # (Dilawer 2026-08-04). Sampled far apart on a perceptual ramp.
        import matplotlib.cm as _cm
        shade = _cm.get_cmap("viridis")(0.15 + 0.7 * j / max(len(all_targets) - 1, 1))
        ax.plot(range(len(lrs)), ys, marker="o", ms=7, lw=2.0,
                color=shade, label=t)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:g}" for lr in lrs], fontsize=S.SIZE["tick"])
    ax.set_xlabel("learning rate", fontsize=S.SIZE["label"])
    ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
    # A fixed window makes the flatness the point: five orders of magnitude of
    # learning rate move the result by less than a decibel.
    lo = min(v for m in d["lr"].values() for v in m.values())
    hi = max(v for m in d["lr"].values() for v in m.values())
    ax.set_ylim(lo - 3, hi + 3)
    # Say what the probe actually is: 5 rates spanning FOUR decades, on two
    # targets at 1000 epochs, not the 18-target 10,000-epoch protocol of (a).
    ax.set_title("Learning rate over four decades:\n"
                 "under one decibel of movement", fontsize=S.SIZE["title"],
                 pad=10)
    ax.text(0.5, -0.30, f"{len(all_targets)} targets, 1000 epochs",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=S.SIZE["tick_small"], color=S.COLOR_DIAG, style="italic")
    ax.legend(frameon=False, fontsize=S.SIZE["legend"], loc="upper right")
    S.style_axis(ax); S.add_panel_label(ax, "b")

    fig.tight_layout(w_pad=2.6)
    S.save(fig, out_path)
    return fig


def save_tables(d: dict, table_dir) -> None:
    """Persist the printed summaries."""
    S.write_table(
        table_dir, "e1_fno_summary",
        ["method"] + S.SUMMARY_HEADER,
        [[S.label(a)] + S.summary_row(d["summary"][a]) for a in ARMS],
        note="E1 | FNO comparison, simulation, ideal physics, PSNR (dB), "
             "n=18 targets. Source: sail_scored.json. The narrative quotes "
             "the median.")
    lr_targets = sorted({t for v in d["lr"].values() for t in v})
    S.write_table(
        table_dir, "e1_fno_lr_probe",
        ["learning_rate", "mean_psnr"] + lr_targets,
        [[lr, float(np.mean(list(d["lr"][lr].values())))]
         + [d["lr"][lr].get(t, float("nan")) for t in lr_targets]
         for lr in sorted(d["lr"], reverse=True)],
        note="E1 | FNO learning-rate probe (width 8, modes 500, 1000 "
             "epochs). Hyperparameter study, outside the canonical "
             "dataset; supports the fairness argument only.")


def build(out_dir, path=None, lr_path=None):
    d = compute(path, lr_path)
    report(d)
    save_tables(d, Path(out_dir).parent / "tables")
    fig = draw(d, Path(out_dir) / "e1_fno")
    return fig
