r"""
fig_attention.py -- E3. What the attention is doing, and the bias-dominance
control.

Three panels:

(a) ATTENTION MAPS. The 4x4 token-attention matrix (p = 500 on 1000x1000
    gives 2x2 = 4 tokens), averaged over heads, layers and all 18 targets,
    one map per training regime. Under per-target simulation training the
    maps sit near the uniform value 0.25: with one target memorisable,
    isotropic mixing suffices. Once one model must serve 18 targets through
    real optics, heads commit: rows approach one-hot. Same architecture, same
    data format; the task is what recruits the attention.

(b) BIAS-DOMINANCE CONTROL. Evaluating the trained network with the
    input-dependent pathway removed, so only the learned bias reaches the
    output, collapses reconstruction from 49.1 dB to 8.4 dB (per-target) and
    from 27.4 to 8.4 dB (batched), 18 of 18 targets in both regimes. The
    quality lives in the input-dependent computation, not in a memorised
    offset. SIMULATION-ONLY by design: the diagnostic evaluates checkpoints
    through the simulated forward model, so its sail_* rows are artefacts
    and are deliberately not drawn.

(c) THE SAME MAPS, TARGET BY TARGET. Panel (a) averages over the 18
    targets, which invites the question of whether the average hides the
    story. It does not. One row per training regime, one small map per
    target, on the shared scale of panel (a). Per-target training stays
    near-uniform on every target and batched SAIL is selective on every
    target, so the regime, not the picture, sets the behaviour.

THE TRAP THIS FIGURE PRE-EMPTS. diagnostics.json also records the attention
block's output at roughly 5e-5 of the residual stream norm. Read alone, that
invites "attention barely contributes". Panel (b) is the answer, printed
beside the maps: small in norm, decisive in function.

Reads multilevel/diagnostics/{regime}/{target}/attn.npy (layers, heads, 4, 4)
and diagnostics.json's per-target bias block. Colour: the attention result
owns ACCENT; maps use a single-hue sequential ramp; bias-only bars are neutral
slate, matching the FNO convention for not-ours reference arms.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sailrev as S

REGIMES = [
    ("sim_per_target",    "Per-target,\nsimulation"),
    ("sail_per_target",   "Per-target,\nSAIL"),
    ("sim_batched",       "Batched,\nsimulation"),
    # "(2000)" removed from display (caption states the epoch budget), same
    # decision as the E8 panels.
    ("sail_batched_2000", "Batched,\nSAIL"),
]
BIAS_REGIMES = [("sim_per_target", "Per-target"), ("sim_batched", "Batched")]
UNIFORM = 0.25
SLATE = "#37474f"


def _diag_root() -> Path:
    return (Path(os.environ["SAILREV_RESULTS"]) / "Self-Attention" /
            "multilevel" / "diagnostics")


def mean_attention(regime: str) -> tuple[np.ndarray, int]:
    """Head-, layer- and target-averaged 4x4 attention for one regime."""
    root = _diag_root() / regime
    maps = [np.load(p).mean(axis=(0, 1))
            for p in sorted(root.glob("*/attn.npy"))]
    if not maps:
        raise FileNotFoundError(f"no attn.npy under {root}")
    return np.mean(maps, axis=0), len(maps)


def bias_values(regime: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads((_diag_root() / "diagnostics.json").read_text())
    tg = d["regimes"][regime]
    full = np.array([t["bias"]["psnr_full"] for t in tg.values()])
    bias = np.array([t["bias"]["psnr_bias_only"] for t in tg.values()])
    return full, bias


def save_tables(table_dir) -> None:
    """Persist the bias-dominance control. Simulation regimes only, by
    design: the diagnostic evaluates trained checkpoints through the
    simulated forward model, so the sail_* rows are artefacts and stay
    out."""
    header = (["regime",
               "full_median", "full_mean", "full_sd",
               "bias_median", "bias_mean", "bias_sd",
               "delta_median", "delta_mean", "wins", "n"])
    rows = []
    for key, name in BIAS_REGIMES:
        full, bias = bias_values(key)
        delta = full - bias
        rows.append([name,
                     float(np.median(full)), float(full.mean()),
                     float(full.std(ddof=1)),
                     float(np.median(bias)), float(bias.mean()),
                     float(bias.std(ddof=1)),
                     float(np.median(delta)), float(delta.mean()),
                     int((delta > 0).sum()), int(delta.size)])
    S.write_table(
        table_dir, "e3_bias_control_summary", header, rows,
        note="E3 | bias-dominance control, simulation only, PSNR (dB); "
             "full model vs learned-bias-only evaluation, delta = full "
             "minus bias-only, paired by target. Source: diagnostics.json. "
             "The narrative quotes the median.")


def build(out_dir):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    S.apply_style()

    fig = plt.figure(figsize=(14.0, 5.0), layout="constrained")
    gs = fig.add_gridspec(1, len(REGIMES) + 1,
                          width_ratios=[1] * len(REGIMES) + [1.6],
                          wspace=0.08)

    # ---- (a) attention maps ------------------------------------------------
    vmax = 0.0
    data = []
    for key, _ in REGIMES:
        try:
            m, n = mean_attention(key)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            m, n = np.full((4, 4), np.nan), 0
        data.append((m, n))
        if np.isfinite(m).any():
            vmax = max(vmax, np.nanmax(m))

    ims = None
    map_axes = []
    for i, ((key, title), (m, n)) in enumerate(zip(REGIMES, data)):
        ax = fig.add_subplot(gs[0, i])
        map_axes.append(ax)
        ims = ax.imshow(m, cmap="Purples", vmin=0.0, vmax=vmax,
                        interpolation="nearest")
        for (r, c), v in np.ndenumerate(m):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=S.SIZE["tick_small"],
                        color="white" if v > 0.6 * vmax else "#212121")
        ax.set_title(title + (f"\n(n={n})" if n else "\n(missing)"),
                     fontsize=S.SIZE["title"], pad=8)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"k{j}" for j in range(4)],
                           fontsize=S.SIZE["tick_small"])
        ax.set_yticklabels([f"q{j}" for j in range(4)],
                           fontsize=S.SIZE["tick_small"])
        ax.tick_params(length=0)
        if i == 0:
            S.add_panel_label(ax, "a")

    cb = fig.colorbar(ims, ax=map_axes, orientation="horizontal",
                      fraction=0.05, pad=0.06, shrink=0.55)
    cb.set_label("mean attention weight", fontsize=S.SIZE["tick_small"])
    cb.ax.tick_params(labelsize=S.SIZE["tick_small"])
    UNIFORM_GREEN = "#1b8a4b"   # readable on the light end of the ramp
    cb.ax.axvline(UNIFORM, color=UNIFORM_GREEN, lw=2.2)
    cb.ax.text(UNIFORM, 1.35, "uniform (0.25)", color=UNIFORM_GREEN,
               fontsize=S.SIZE["tick_small"], ha="center", va="bottom",
               transform=cb.ax.get_xaxis_transform())

    # ---- (b) bias-dominance control ---------------------------------------
    ax = fig.add_subplot(gs[0, len(REGIMES)])
    rng = np.random.default_rng(0)
    for i, (key, name) in enumerate(BIAS_REGIMES):
        full, bias = bias_values(key)
        for j, (v, c) in enumerate(((full, S.COLOR_TRANSFORMER),
                                    (bias, SLATE))):
            x = i * 2.2 + j * 0.9
            ax.bar(x, v.mean(), width=0.72, color=c, alpha=0.85, zorder=2)
            ax.scatter(x + rng.uniform(-0.16, 0.16, v.size), v, s=14,
                       color="#212121", alpha=0.5, zorder=3, linewidths=0)
            # No in-panel numbers (they collided with the legend); the values
            # are stated in the caption, and the build prints them for it.
            # The narrative quotes medians and T1 carries means beside
            # them, so BOTH are printed here.
            print(f"  E3b caption value: {name} "
                  f"{'full' if j == 0 else 'bias-only'} = "
                  f"median {np.median(v):.2f} dB (mean {v.mean():.2f})")
        delta = full - bias
        wins = int((delta > 0).sum())
        print(f"  E3b paired delta ({name}): full - bias-only = "
              f"median {np.median(delta):+.2f} dB (mean {delta.mean():+.2f}), "
              f"full better on {wins}/{delta.size}")
    ax.set_xticks([0.45, 2.65])
    ax.set_xticklabels([n for _, n in BIAS_REGIMES], fontsize=S.SIZE["label"])
    ax.set_ylabel("PSNR (dB), simulation", fontsize=S.SIZE["label"])
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=S.COLOR_TRANSFORMER, label="full model"),
        mpatches.Patch(color=SLATE, label="learned bias only")],
        frameon=False, fontsize=S.SIZE["legend"], loc="upper right")
    ax.set_title("Remove the input-dependent pathway\nand quality collapses",
                 fontsize=S.SIZE["title"], pad=8)
    S.style_axis(ax)
    S.add_panel_label(ax, "b")

    save_tables(Path(out_dir).parent / "tables")
    S.save(fig, Path(out_dir) / "e3_attention")

    # ---- (c) per-target small multiples. Panel (a) averaged over targets,
    # so this shows the same maps target by target and answers "is the
    # average hiding it". One row per regime, 18 tiny maps per row, on
    # panel (a)'s scale. No numbers: the pattern is the point.
    import matplotlib.pyplot as plt2
    targets = sorted(p.parent.name for p in
                     (_diag_root() / REGIMES[0][0]).glob("*/attn.npy"))
    # Cell height is matched to cell width (both ~0.74 in after margins) so
    # equal wspace/hspace fractions give visually equal gaps; previously the
    # taller rows turned the same fraction into a larger row gap.
    fig2, axes2 = plt.subplots(len(REGIMES), max(len(targets), 1),
                               figsize=(0.78 * max(len(targets), 1) + 1.6,
                                        0.85 * len(REGIMES)),
                               squeeze=False)
    for r, (key, title) in enumerate(REGIMES):
        for c, t in enumerate(targets):
            ax2 = axes2[r][c]
            p = _diag_root() / key / t / "attn.npy"
            if p.exists():
                ax2.imshow(np.load(p).mean(axis=(0, 1)), cmap="Purples",
                           vmin=0.0, vmax=vmax, interpolation="nearest")
            ax2.set_xticks([]); ax2.set_yticks([])
            # No per-column target names (Dilawer): the pattern is the
            # point, and 18 rotated labels were noise.
            if c == 0:
                ax2.set_ylabel(title.replace("\n", " "),
                               fontsize=S.SIZE["tick_small"], rotation=0,
                               ha="right", va="center", labelpad=6)
                if r == 0:
                    S.add_panel_label(ax2, "c", dx=-96, dy=26)
    fig2.suptitle("Attention maps for every target, by training regime",
                  fontsize=S.SIZE["title_wide"])
    fig2.subplots_adjust(wspace=0.05, hspace=0.05, left=0.14, right=0.99,
                         top=0.88, bottom=0.03)
    S.save(fig2, Path(out_dir) / "e3c_attention_maps_by_target")
    return fig
