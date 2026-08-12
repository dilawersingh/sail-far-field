r"""
fig_batched.py -- Fig 4. One model, eighteen targets.

THE GENERALISATION LEAD, and the claim no iterative method can match for
structural rather than contingent reasons. Three panels, three separate
assertions, each measured:

  (a) PARITY. A single shared model trained jointly across all 18 targets
      reaches the quality of 18 individually trained SAIL models. Paired
      per-target differences at both epoch budgets. The distinction between
      per-target optimisation and generalisation is that the shared model
      needs neither.

  (b) INFERENCE COST, WITHIN THE TRAINED SET. What it takes to obtain a
      hologram for one of the eighteen targets. The shared model runs one
      forward pass; every other method must run a full camera-in-the-loop
      optimisation for each target, every time. Log scale: five orders of
      magnitude.

      SCOPE, STATED ON THE FIGURE ITSELF: this is amortisation across the set
      the model was trained on, NOT generalisation to unseen targets. Calling
      the bottom bar "a new target" would overclaim; the unseen-target study
      is E6, at its own resolution.

  (c) ATTENTION SELECTIVITY. THE HEADLINE. Deviation from uniform mixing rises
      as the task demands generalisation: near-isotropic when one model serves
      one target, strongly selective when one model must serve eighteen. This
      is the mechanism behind (a), and it spans the full width of the figure
      for that reason.

SCOPE: INFERENCE, NOT TRAINING. This figure talks only about the cost of
serving a target. Training wall-clock is reported in T2 and Methods; it is
deliberately absent here, because a figure carries one claim. Internal guardrail, so it is never
reintroduced by accident: batched training is NOT cheaper than per-target
training (both spend one camera exposure per target per epoch), so any future
edit that puts training time on this figure would be making a claim the logs
contradict.

WHY 750 AND 2000 ARE BOTH SHOWN. Parity at 750 could be dismissed as an
artefact of stopping early. At 2000 the shared model slightly exceeds
per-target SAIL, and at 750 it sits slightly below. Showing only the better
budget would be selective reporting.

Panel (c) reads multilevel/diagnostics/diagnostics.json, which is outside
sail_scored.json because it measures attention weights rather than image
quality. It is the one input in this figure not drawn from the canonical
dataset, and the loader states its provenance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sailrev as S

# Attention regimes, in order of how much the task demands of the model:
# one model one target, then hardware, then one model eighteen targets.
REGIMES = [("sim_per_target", "1 target\nsimulation"),
           ("sail_per_target", "1 target\n+ hardware"),
           ("sim_batched", "18 targets\nsimulation"),
           ("sail_batched_2000", "18 targets\n+ hardware")]

# What serving one of the 18 trained targets costs. GD+CITL and per-target SAIL must run a full
# camera-in-the-loop optimisation for it; the shared model already covers it.
# (method, kind, note). kind is a FLAG, not display text: "inference" means the
# model already covers the target and the cost is one forward pass; "optimise"
# means a fresh camera-in-the-loop run. Matching on display text is what broke
# this once, so the flag and the caption are separate fields.
COST_ROWS = [("gd_citl_random", "optimise", "camera optimisation per target"),
             ("gd_citl_warm", "optimise", "camera optimisation per target"),
             ("sail", "optimise", "camera optimisation per target, own model"),
             ("batched_sail_2000", "inference", "one forward pass, shared model")]
# Short names for panel (b): the full LABEL strings crowd the neighbouring
# panel, and the row's meaning is carried by the annotation beneath it.
# Two lines, not one: a single-line "Batched SAIL (shared)" is wide enough to
# reach into panel (a)'s data area no matter how much gutter is added.
COST_SHORT = {"gd_citl_random": "GD+CITL\nrandom",
              "gd_citl_warm": "GD+CITL\nsim-seeded",
              "sail": "SAIL\nper target",
              "batched_sail_2000": "Batched SAIL\nshared"}
INFERENCE_SECONDS = 0.015      # Ext Fig 1D; re-measure before quoting


def diagnostics_path() -> Path:
    if os.environ.get("SAILREV_DIAGNOSTICS"):
        return Path(os.environ["SAILREV_DIAGNOSTICS"])
    if os.environ.get("SAILREV_SA"):
        return Path(os.environ["SAILREV_SA"]) / "diagnostics" / "diagnostics.json"
    if os.environ.get("SAILREV_RESULTS"):
        return (Path(os.environ["SAILREV_RESULTS"]) / "Self-Attention" /
                "multilevel" / "diagnostics" / "diagnostics.json")
    raise RuntimeError("set SAILREV_DIAGNOSTICS or SAILREV_RESULTS in Cell 0")


def load_attention(path=None) -> dict:
    """{regime: {'mean': float, 'per_target': {target: deviation}}}.

    deviation = 1 - H(attention)/log(T): 0 is isotropic mixing, 1 is a fully
    selective head. Computed by architecture_diagnostics.py from the saved
    attention maps, not recomputed here.
    """
    d = json.loads(Path(path or diagnostics_path()).read_text())
    out = {}
    for regime, per_target in d["regimes"].items():
        out[regime] = {
            "mean": d["summary_deviation"][regime],
            "per_target": {t: v["deviation_mean"] for t, v in per_target.items()},
        }
    return out


def compute(path=None, diag_path=None) -> dict:
    parity = {}
    for physics in S.PHYSICS:
        parity[physics] = {
            k: S.delta_summary("bench", physics, k, "sail", "psnr", path)
            for k in ("batched_sail_750", "batched_sail_2000")}
        for k in parity[physics]:
            _, d = S.paired("bench", physics, k, "sail", "psnr", path)
            parity[physics][k]["deltas"] = d

    T = {}
    for t in S.load(path)["timings"]:
        if t["seconds"] is not None:
            T.setdefault((t["method"], t["physics"]), []).append(t["seconds"])

    cost = []
    for method, kind, note in COST_ROWS:
        if kind == "inference":
            cost.append({"method": method, "seconds": INFERENCE_SECONDS,
                         "kind": kind, "note": note})
        else:
            v = T.get((method, "ideal"), [])
            if v:
                cost.append({"method": method, "seconds": float(np.median(v)),
                             "kind": kind, "note": note})
    return {"parity": parity, "cost": cost,
            "attention": load_attention(diag_path)}


def report(data: dict) -> None:
    print("\nParity: batched shared model minus per-target SAIL (bench, PSNR dB)")
    for physics in S.PHYSICS:
        for k, r in data["parity"][physics].items():
            print(f"  {S.PHYSICS_LABEL[physics]:14s} {S.label(k):22s} "
                  f"median {r['median']:+5.2f} "
                  f"[{r['ci_lo']:+5.2f},{r['ci_hi']:+5.2f}] "
                  f"wins {r['wins']:2d}/{r['n_pairs']}  p={r['p']:.2e}")

    print("\nInference: one of the 18 targets the model serves (ideal)")
    for c in data["cost"]:
        v = c["seconds"]
        pretty = f"{v:.3f} s" if v < 60 else f"{v/60:.1f} min"
        print(f"  {S.label(c['method']):24s} {pretty:>10s}   {c['note']}")
    slow = max(c["seconds"] for c in data["cost"])
    fast = min(c["seconds"] for c in data["cost"])
    print(f"  ratio: {slow/fast:,.0f}x")

    print("\nAttention deviation from uniform (0 isotropic, 1 fully selective)")
    for regime, lab in REGIMES:
        a = data["attention"][regime]
        print(f"  {lab.replace(chr(10),' '):26s} {a['mean']:.3f}")
    a = data["attention"]
    print(f"  control, 750 vs 2000 epochs: {a['sail_batched_750']['mean']:.3f} "
          f"vs {a['sail_batched_2000']['mean']:.3f}  "
          f"(selectivity is not a training-length artefact)")


def draw(data: dict, out_path):
    """Layout: (a) and (b) share the top row, (c) spans the bottom.

    Panel (c) is the paper's novel result, so it gets the full width and the
    largest marks. The two supporting panels sit above it.
    """
    import matplotlib.pyplot as plt
    S.apply_style()

    fig = plt.figure(figsize=(13.6, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], hspace=0.40,
                          wspace=0.42, width_ratios=[1.12, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    rng = np.random.default_rng(0)

    # ---- (a) parity ------------------------------------------------------
    ax = ax_a
    ax.axvline(0, color=S.COLOR_DIAG, lw=1.0, ls="--", zorder=1)
    # Grouped by epoch budget, not by forward model: the comparison a reader
    # makes is 750 against 2000, so those rows sit together and the two physics
    # conditions pair within each budget.
    rows = [(physics, k) for k in ("batched_sail_750", "batched_sail_2000")
            for physics in S.PHYSICS]
    for i, (physics, k) in enumerate(rows):
        r = data["parity"][physics][k]
        y = len(rows) - 1 - i
        c = S.method_color(k)
        ax.scatter(r["deltas"], y + rng.uniform(-0.13, 0.13, r["deltas"].size),
                   s=24, color=c, alpha=0.45, edgecolors="none", zorder=2)
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2.6,
                solid_capstyle="round", zorder=3)
        ax.plot([r["median"]], [y], "o", ms=9, color=c,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{k.split('_')[-1]} epochs, {p}"
                        for p, k in rows][::-1], fontsize=S.SIZE["annot"])
    ax.set_xlabel("Δ PSNR (dB) vs per-target SAIL", fontsize=S.SIZE["label"])
    # The 750 rows are significantly BELOW per-target SAIL (1/18 and 2/18
    # wins). A title claiming parity is contradicted by half its own panel.
    ax.set_title("Shared model: at parity by 2000 epochs,\n"
                 "0.3-0.4 dB below at 750", fontsize=S.SIZE["title"], pad=10)
    S.style_axis(ax); S.add_panel_label(ax, "a")

    # ---- (b) inference cost for a new target -----------------------------
    ax = ax_b
    cost = data["cost"]
    ys = np.arange(len(cost))[::-1]
    for y, c in zip(ys, cost):
        col = S.method_color(c["method"])
        ax.barh(y, c["seconds"], height=0.55, color=col,
                edgecolor="white", linewidth=1.5)
        v = c["seconds"]
        ax.text(v * 1.6, y, f"{v:.3f} s" if v < 60 else f"{v/60:.0f} min",
                va="center", ha="left", fontsize=S.SIZE["legend"], color=col,
                fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 3e5)
    ax.set_yticks(ys)
    ax.set_ylim(-0.6, len(cost) - 0.4)
    ax.set_yticklabels([COST_SHORT.get(c["method"], S.label(c["method"]))
                        for c in cost], fontsize=S.SIZE["tick"])
    # No floating annotations here: they crowded the origin. The scope lives in
    # the axis label and the row names, where it cannot be missed or cropped.
    ax.set_xlabel("time per target, seconds (log)", fontsize=S.SIZE["label"])
    ax.set_title("Serving one of the 18 targets:\none forward pass, not an "
                 "optimisation", fontsize=S.SIZE["title"], pad=10)
    slow = max(c["seconds"] for c in cost); fast = min(c["seconds"] for c in cost)
    ax.text(0.98, 0.06, f"{slow/fast:,.0f}× faster", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=S.SIZE["label"], fontweight="bold",
            color=S.method_color("batched_sail_2000"))
    S.style_axis(ax); S.add_panel_label(ax, "b")

    # ---- (c) attention selectivity, the headline -------------------------
    ax = ax_c
    att = data["attention"]
    xs = np.arange(len(REGIMES))
    targets = sorted(att[REGIMES[0][0]]["per_target"])
    for t in targets:
        ys_t = [att[r]["per_target"].get(t, np.nan) for r, _ in REGIMES]
        ax.plot(xs, ys_t, color=S.ACCENT_FAINT, alpha=0.55, lw=1.1, zorder=1)
    means = [att[r]["mean"] for r, _ in REGIMES]
    ax.plot(xs, means, color=S.ACCENT, lw=3.2, marker="o", ms=13,
            markeredgecolor="white", markeredgewidth=1.6, zorder=3,
            label="mean over 18 targets")
    for x, m in zip(xs, means):
        below = m < 0.3
        ax.text(x, m - 0.055 if below else m + 0.045, f"{m:.3f}",
                ha="center", va="top" if below else "bottom", fontsize=S.SIZE["annot_big"],
                color=S.ACCENT, fontweight="bold")

    # The two regimes on the left share one model per target; the two on the
    # right make a single model serve all eighteen. Naming that split is what
    # makes the rise legible without reading the caption.
    # Both notes sit clear of the traces: the left cluster runs near zero so
    # its note goes above; the right cluster bottoms out around 0.31, so its
    # note goes BELOW rather than into the fan of lines.
    ax.annotate("one model per target", xy=(0.5, 0.30), xycoords="data",
                ha="center", fontsize=S.SIZE["label"], style="italic",
                color=S.COLOR_DIAG)
    ax.annotate("one model, 18 targets", xy=(2.5, 0.20), xycoords="data",
                ha="center", fontsize=S.SIZE["label"], style="italic",
                color=S.COLOR_DIAG)

    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in REGIMES], fontsize=S.SIZE["tick"])
    ax.set_xlim(-0.45, len(REGIMES) - 0.55)
    ax.set_ylim(-0.10, 1.06)
    # Simplified per Dilawer 2026-08-04. The definition (1 - H/log T, i.e.
    # 0 = uniform mixing, 1 = fully selective; Shannon entropy of the
    # attention rows) moves to the CAPTION, where it can be stated properly.
    ax.set_ylabel("Attention selectivity", fontsize=S.SIZE["label"])
    ax.set_title("Self-attention becomes selective as the task demands "
                 "generalisation", fontsize=S.SIZE["title_wide"], pad=12)
    ax.legend(frameon=False, fontsize=S.SIZE["legend"], loc="upper left")
    S.style_axis(ax); S.add_panel_label(ax, "c")

    S.save(fig, out_path)
    return fig


def build(out_dir, path=None, diag_path=None):
    data = compute(path, diag_path)
    report(data)
    fig = draw(data, Path(out_dir) / "fig4_one_model_eighteen_targets")
    print("\nCaption discipline. (b) is amortisation ACROSS THE TRAINED SET, not "
          "generalisation\nto unseen targets: the shared model serves any of its 18 "
          "in one forward pass, while\nevery other method re-optimises per target. "
          "Do not write \"a new target\". Training\nwall-clock belongs in T2 and "
          "Methods (R2.2), not here. GS and GD have no shared\nweights, so they "
          "have no counterpart to the shared-model row at all.")
    return fig
