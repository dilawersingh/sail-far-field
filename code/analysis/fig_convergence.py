r"""
fig_convergence.py -- E5. Convergence and compute.

GS and GD swept to 10,000 iterations, the transformer's own training
budget, under both forward models. Every line, the dashed transformer
level, the crossover and the wall-clock abscissa are medians over the 18
targets. Shading is the 25th-75th percentile band in all four panels; in
(b)/(d) it is drawn at the median-time abscissa, because each target
reaches a given budget at a slightly different time.

Each point is an independent run from a random initialisation rather than
a checkpoint along one trajectory, so the scatter between adjacent points
includes run-to-run variation.

The wall-clock panels show both transformer costs: the filled marker is
one forward pass, and the open marker adds the per-target training that
produced the checkpoint.

Reads the sweep's own all_results.json under each simulation-comparison
tree; the operating-point values match sail_scored.json.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import sailrev as S

# The operating point: the iteration count used by Fig 2 and by every bench
# comparison. Marked on the sweep so a reader sees exactly where the paper's
# headline GS/GD numbers sit on the full convergence curve, which is R2's
# stopped-early question answered visually. (Formerly labelled "reported: 750",
# which became ambiguous once the 10k sweep was itself reported.)
REPORTED_ITER = 750


def sweep_path(physics: str, root=None, suffix: str = "_10k") -> Path:
    """Locate a sweep aggregate. Falls back to the pre-10k sweep if absent.

    The 10k sweep writes to simulation_comparison_{physics}_10k under the
    Self-Attention tree; the original 750/1500 sweep lives under GD+CITL. This
    prefers the newer one and says which it used, so a figure built before the
    long run finished is never mistaken for one built after.
    """
    if root:
        return Path(root)
    env = os.environ.get(f"SAILREV_SWEEP_{physics.upper()}")
    if env:
        return Path(env)
    results = Path(os.environ["SAILREV_RESULTS"])
    candidates = [
        results / "Self-Attention" / "multilevel" / "simulations" /
        f"simulation_comparison_{physics}{suffix}",
        results / "GD+CITL" / "simulations" / f"simulation_comparison_{physics}",
    ]
    for base in candidates:
        if base.exists():
            hits = sorted(base.glob("*_all_results.json"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"no sweep aggregate for physics={physics}; looked in "
        + ", ".join(str(c) for c in candidates))


def load_sweep(physics: str, root=None, formulation: str = "amplitude") -> dict:
    """{method: {iterations: [psnr per target]}} plus wall-clock, from one sweep.

    formulation="amplitude" reads regime1 directly (the published formulation,
    written first by the runner). Any other value reads the namespaced sibling
    under regime1["additional_target_formulations"], which carries the sweeps
    but no transformer entry: the transformer is trained once, on the amplitude
    target, so its number is taken from regime1 in either case.
    """
    path = sweep_path(physics, root)
    rows = json.loads(Path(path).read_text())
    psnr = {m: defaultdict(list) for m in ("gs", "gd")}
    secs = {m: defaultdict(list) for m in ("gs", "gd")}
    t_psnr, t_infer, t_train = [], [], []
    for r in rows:
        reg = r["summary"]["regime1"]
        block = reg if formulation == "amplitude" else \
            reg.get("additional_target_formulations", {}).get(formulation, {})
        for m in ("gs", "gd"):
            for e in block.get(f"{m}_sweep", []):
                psnr[m][int(e["iterations"])].append(float(e["psnr"]))
                secs[m][int(e["iterations"])].append(float(e["wall_clock_seconds"]))
        t = reg.get("transformer")
        if isinstance(t, dict) and "psnr" in t:
            t_psnr.append(float(t["psnr"]))
            if "wall_clock_seconds" in t:
                t_infer.append(float(t["wall_clock_seconds"]))
        if r.get("training_seconds"):
            t_train.append(float(r["training_seconds"]))
    return {"path": str(path), "formulation": formulation,
            "psnr": psnr, "seconds": secs,
            "transformer": t_psnr, "t_infer": t_infer, "t_train": t_train,
            "n_targets": len(rows)}


def compute(root_ideal=None, root_faithful=None,
            formulation: str = "amplitude") -> dict:
    out = {}
    for physics, root in (("ideal", root_ideal), ("faithful", root_faithful)):
        try:
            out[physics] = load_sweep(physics, root, formulation)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [skip] {physics}: {e}")
    return out


def report(d: dict) -> None:
    for physics, s in d.items():
        iters = sorted(s["psnr"]["gd"])
        print(f"\nE5 | convergence ({S.PHYSICS_LABEL[physics]}, "
              f"{s['formulation']} target, n={s['n_targets']} targets)")
        print(f"  source: {s['path']}")
        print(f"  {'iter':>6s} {'GS med':>9s} {'GD med':>9s} {'GD gain':>9s} "
              f"{'GD s/run':>9s}")
        prev = None
        for it in iters:
            gs = np.median(s["psnr"]["gs"][it]) if s["psnr"]["gs"].get(it) else np.nan
            gd = np.median(s["psnr"]["gd"][it])
            sec = np.median(s["seconds"]["gd"][it])
            gain = f"{gd - prev:+9.2f}" if prev is not None else " " * 9
            prev = gd
            print(f"  {it:6d} {gs:9.2f} {gd:9.2f} {gain} {sec:9.2f}")

        if s["transformer"]:
            t = float(np.median(s["transformer"]))
            line = f"  transformer: {t:.2f} dB"
            if s["t_infer"]:
                line += f" in {np.median(s['t_infer']) * 1e3:.1f} ms (one forward pass)"
            if s["t_train"]:
                line += f"; per-target training {np.median(s['t_train']) / 60:.1f} min"
            print(line)
            if iters:
                best = float(np.median(s["psnr"]["gd"][iters[-1]]))
                print(f"  gap at {iters[-1]} iterations: transformer leads GD by "
                      f"{t - best:+.2f} dB")

        # The question the whole sweep exists to answer, stated numerically.
        # Three outcomes, not two. The crossover case was the one that landed,
        # and it is the one a two-way verdict would have silently mislabelled.
        if len(iters) >= 2:
            last, prev_it = iters[-1], iters[-2]
            gd_last = float(np.median(s["psnr"]["gd"][last]))
            tail = gd_last - float(np.median(s["psnr"]["gd"][prev_it]))
            flat = abs(tail) < 0.5
            t = float(np.median(s["transformer"])) if s["transformer"] else None
            if t is not None and gd_last > t:
                x = crossover(s, t)
                where = f" at about {x} iterations" if x else ""
                verdict = f"GD overtakes the transformer{where}."
            elif flat:
                verdict = ("GD has converged below the transformer level.")
            else:
                verdict = ("GD is below the transformer and still climbing "
                           "at the end of the sweep.")
            print(f"  final step {prev_it}->{last}: {tail:+.2f} dB  -> {verdict}")


def crossover(s: dict, t: float):
    """First iteration count at which median GD overtakes the transformer."""
    for i in sorted(s["psnr"]["gd"]):
        if float(np.median(s["psnr"]["gd"][i])) > t:
            return i
    return None


def _series(s, method, key_x="iterations"):
    """Aligned (x, y) for one method, dropping any iteration count missing either."""
    iters = sorted(i for i in s["psnr"][method]
                   if s["psnr"][method].get(i) and s["seconds"][method].get(i))
    ys = [float(np.median(s["psnr"][method][i])) for i in iters]
    if key_x == "iterations":
        return iters, ys
    return [float(np.median(s["seconds"][method][i])) for i in iters], ys


def draw(d: dict, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    S.apply_style()

    physics_present = [p for p in S.PHYSICS if p in d]
    nrow = len(physics_present)
    # Rows by physics (a,b ideal / c,d faithful), per Dilawer 2026-08-04:
    # left column quality-vs-iterations, right column quality-vs-wall-clock.
    fig, axes = plt.subplots(nrow, 2, figsize=(14.4, 5.4 * nrow),
                             squeeze=False)

    letters = "abcd"
    for row, physics in enumerate(physics_present):
        s = d[physics]
        iters = sorted(s["psnr"]["gd"])
        c_t = S.method_color("transformer_per_target")
        t = float(np.median(s["transformer"])) if s["transformer"] else None
        gd_max = max(float(np.median(s["psnr"]["gd"][i]))
                     for i in s["psnr"]["gd"])

        # ---- left: quality vs iterations ------------------------------
        ax = axes[row][0]
        for m in ("gs", "gd"):
            xs, ys = _series(s, m, "iterations")
            lo = [np.percentile(s["psnr"][m][i], 25) for i in xs]
            hi = [np.percentile(s["psnr"][m][i], 75) for i in xs]
            c = S.method_color(m)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.16, linewidth=0)
            ax.plot(xs, ys, color=c, lw=2.6, marker="o", ms=6,
                    markeredgecolor="white", markeredgewidth=1.0)
        ax.set_xscale("log")
        if t is not None:
            ax.axhline(t, color=c_t, lw=2.2, ls="--", zorder=3)
            ax.set_ylim(top=max(t, gd_max) + 4.5)
            # Label ABOVE the line; offset in POINTS so a and c sit at the
            # same visual distance despite very different y-ranges.
            ax.annotate(f"{t:.2f} dB", xy=(0.02, t),
                        xycoords=ax.get_yaxis_transform(),
                        xytext=(0, 6), textcoords="offset points",
                        color=c_t, fontsize=S.SIZE["annot"],
                        fontweight="bold", va="bottom", ha="left",
                        bbox=S.halo())
            x = crossover(s, t)
            if x:
                ax.plot([x], [float(np.median(s["psnr"]["gd"][x]))], marker="o",
                        ms=15, mfc="none", color=S.COLOR_DIAG, mew=2.0,
                        linestyle="none", zorder=6)
                ax.annotate(f"GD overtakes\nby {x:,} iterations",
                            xy=(x, float(np.median(s["psnr"]["gd"][x]))),
                            xytext=(12, -16), textcoords="offset points",
                            ha="left", va="top", fontsize=S.SIZE["annot"],
                            color="#212121", bbox=S.halo())
        # reported operating point: label centered ON the dotted line.
        ax.axvline(REPORTED_ITER, color=S.COLOR_DIAG, lw=1.0, ls=":")
        # Horizontal, near the top, centered on the line, and the dotted line
        # is allowed to cross it (no halo) -- Dilawer 2026-08-04.
        ax.text(REPORTED_ITER, 0.955, "operating point: 750",
                va="center", ha="center", fontsize=S.SIZE["tick_small"],
                color=S.COLOR_DIAG,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel("iterations (log)", fontsize=S.SIZE["label"])
        ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
        ax.set_title(f"{S.PHYSICS_LABEL[physics]}: quality vs iterations",
                     fontsize=S.SIZE["title"], pad=10)
        S.style_axis(ax)
        S.add_panel_label(ax, letters[row * 2])

        # ---- right: quality vs wall-clock -----------------------------
        ax = axes[row][1]
        for m in ("gs", "gd"):
            xs, ys = _series(s, m, "seconds")
            # PSNR IQR band, drawn at the median-time abscissa (Dilawer
            # 2026-08-05, matching panels a/c). The caption states: shading
            # is the PSNR IQR; x is the median wall-clock per target, since
            # each target reaches a given budget at a slightly different
            # time.
            it_aligned = sorted(i for i in s["psnr"][m]
                                if s["psnr"][m].get(i)
                                and s["seconds"][m].get(i))
            lo = [np.percentile(s["psnr"][m][i], 25) for i in it_aligned]
            hi = [np.percentile(s["psnr"][m][i], 75) for i in it_aligned]
            c = S.method_color(m)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.16, linewidth=0)
            ax.plot(xs, ys, color=c, lw=2.6, marker="o",
                    ms=6, markeredgecolor="white", markeredgewidth=1.0)
        ax.set_xscale("log")
        if t is not None:
            ax.axhline(t, color=c_t, lw=2.0, ls="--", zorder=2, alpha=0.55)
            ax.set_ylim(top=max(t, gd_max) + 4.5)
            # dB value on the dashed line, centre-left, clear of the circle.
            ax.annotate(f"{t:.2f} dB", xy=(0.45, t),
                        xycoords=ax.get_yaxis_transform(),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", color=c_t,
                        fontsize=S.SIZE["annot"], fontweight="bold",
                        bbox=S.halo())
            if s["t_infer"]:
                # Circle ON the dashed line at the forward-pass time, darker
                # purple, horizontal two-line label -- Dilawer 2026-08-04.
                x_i = float(np.median(s["t_infer"]))
                dark = S.PURPLE_F[1]
                ax.plot([x_i], [t], marker="o", ms=14, mfc="none",
                        color=dark, mew=2.4, linestyle="none", zorder=6)
                ax.annotate(f"Transformer\nforward pass "
                            f"({x_i * 1e3:.0f} ms)",
                            xy=(x_i, t), xytext=(10, -12),
                            textcoords="offset points", ha="left", va="top",
                            color=dark, fontsize=S.SIZE["tick_small"],
                            fontweight="bold", bbox=S.halo())
                ax.set_xlim(left=x_i * 0.55)
        ax.set_xlabel("wall-clock seconds per target (log)",
                      fontsize=S.SIZE["label"])
        ax.set_ylabel("PSNR (dB)", fontsize=S.SIZE["label"])
        ax.set_title(f"{S.PHYSICS_LABEL[physics]}: quality vs compute",
                     fontsize=S.SIZE["title"], pad=10)
        # Shared y within the row (a with b, c with d): the dashed transformer
        # line then sits at the same height across the pair.
        ax.set_ylim(axes[row][0].get_ylim())
        S.style_axis(ax)
        S.add_panel_label(ax, letters[row * 2 + 1])

    # One shared legend for the whole figure (was one per panel).
    handles = [Line2D([], [], color=S.method_color("gs"), lw=2.6, marker="o",
                      ms=6, markeredgecolor="white", label=S.label("gs")),
               Line2D([], [], color=S.method_color("gd"), lw=2.6, marker="o",
                      ms=6, markeredgecolor="white", label=S.label("gd")),
               Line2D([], [], color=S.method_color("transformer_per_target"),
                      lw=2.2, ls="--", label="Transformer (one forward pass)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=S.SIZE["legend"], bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=3.0, h_pad=3.0)
    S.save(fig, out_path)
    return fig


def build(out_dir, root_ideal=None, root_faithful=None,
          formulation: str = "amplitude"):
    d = compute(root_ideal, root_faithful, formulation)
    if not d:
        print("  no sweep data found; skipping E5")
        return None
    report(d)
    fig = draw(d, Path(out_dir) / "e5_convergence")
    return fig
