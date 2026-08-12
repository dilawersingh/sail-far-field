r"""
sailrev.py -- shared foundation for every revised-manuscript figure and table.

SCRIPT 1 of the analysis pipeline (score_revision.py is script 0). Nothing here
draws a figure; it is the layer every figure script sits on, so loading,
terminology, statistics and visual style are defined exactly once and cannot
drift between figures. Same three principles as gdcitl.py, for the same
reasons:

  1. READ-ONLY, FROM ONE SOURCE. Every quantitative artifact reads
     sail_scored.json and nothing else. No figure is built from a
     hand-edited intermediate, and no figure re-scores anything: if a number is
     not in the canonical JSON it does not go in the paper.

  2. TERMINOLOGY IS ENFORCED HERE, NOT PER-FIGURE. The JSON stores method keys
     (`transformer_per_target`, `gd_citl_random`); the paper has display names
     ("Transformer", "GD+CITL (random init)"). LABEL is the only place that
     mapping exists, so nothing internal reaches a caption.

  3. MEDIANS, NOT MEANS, FOR BENCH CLAIMS. Bench distributions are
     right-skewed across the target set and n = 18. Every headline statistic is
     a median with a seeded bootstrap interval; every test is paired and
     non-parametric. Means and SDs are still reported in T1, so T1 carries
     both and the narrative quotes the median.

Paths: SAILREV_SCORED, else SAILREV_OUT, else <SAILREV_RESULTS>/Self-Attention/
multilevel/analysis/sail_scored.json. Resolved lazily inside load(), so
importing this module never fails for want of an environment variable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
N_TARGETS = 18
PHYSICS = ("ideal", "faithful")


def scored_path() -> Path:
    for var in ("SAILREV_SCORED", "SAILREV_OUT"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("SAILREV_RESULTS"):
        return (Path(os.environ["SAILREV_RESULTS"]) / "Self-Attention" /
                "multilevel" / "analysis" / "sail_scored.json")
    raise RuntimeError(
        "No scored dataset located. Set SAILREV_SCORED (or SAILREV_OUT, or "
        "SAILREV_RESULTS) in the notebook's Cell 0 before importing."
    )


_CACHE: dict | None = None


def load(path=None, refresh: bool = False) -> dict:
    """The canonical dataset: {'meta', 'records', 'timings'}. Cached."""
    global _CACHE
    if _CACHE is None or refresh or path is not None:
        p = Path(path) if path else scored_path()
        _CACHE = json.loads(p.read_text())
        _CACHE["_path"] = str(p)
    return _CACHE


# --------------------------------------------------------------------------
# Domain resolution (decision 2026-08-04, agreement gate PASSED)
#
# The manuscript's bench dataset is the single-alignment re-capture
# (domain "bench_replay" in sail_scored.json: 504 captures, 14 methods,
# one session, one rig state, dust-cleaned). The gate that authorised this:
# same stored holograms photographed 11 days apart reproduce with median
# |delta| 0.14 dB; the identical-array transformer pairs agree to 0.03 dB.
#
# Figure modules keep asking for domain "bench" -- resolved HERE to
# BENCH_DOMAIN, with the replay's gs_750/gd_750 renamed to gs/gd (750 is the
# published operating point, so the display concept is unchanged). Methods
# only the replay has (gs_10000, gd_10000, sail_plus) keep their own names.
# The July-era captures stay readable as domain "bench_july" for provenance
# and the agreement analysis; nothing in the manuscript quotes them.
# --------------------------------------------------------------------------
BENCH_DOMAIN = "bench_replay"
_BENCH_RENAME = {"gs_750": "gs", "gd_750": "gd"}


def records(domain=None, physics=None, method=None, path=None) -> list[dict]:
    rs = load(path)["records"]
    if domain:
        real = {"bench": BENCH_DOMAIN, "bench_july": "bench"}.get(domain, domain)
        rs = [r for r in rs if r["domain"] == real]
        if domain == "bench" and real != "bench":
            rs = [{**r, "method": _BENCH_RENAME.get(r["method"], r["method"])}
                  for r in rs]
    if physics:
        rs = [r for r in rs if r["physics"] == physics]
    if method:
        ms = {method} if isinstance(method, str) else set(method)
        rs = [r for r in rs if r["method"] in ms]
    return rs


def targets(path=None) -> list[str]:
    return sorted({r["target"] for r in load(path)["records"]})


def by_target(domain: str, physics: str, method: str, metric: str = "psnr",
              path=None) -> dict[str, float]:
    out = {r["target"]: r[metric]
           for r in records(domain, physics, method, path)}
    if not out:
        d = load(path)
        have = sorted({(r["domain"], r["physics"], r["method"])
                       for r in d["records"]})
        raise RuntimeError(
            f"no records for domain={domain!r} physics={physics!r} "
            f"method={method!r} in {d['_path']}. The file holds "
            f"{len(d['records'])} records across {len(have)} families; "
            f"families present: {have}")
    return out


def paired(domain: str, physics: str, a: str, b: str, metric: str = "psnr",
           path=None) -> tuple[list[str], np.ndarray]:
    """Per-target (a - b), over targets present in BOTH arms.

    Returns the target list alongside the deltas so a caller can label points;
    pairing is by target name, never by list position, because a missing run in
    one arm would otherwise silently shift every subsequent pair.
    """
    A = by_target(domain, physics, a, metric, path)
    B = by_target(domain, physics, b, metric, path)
    ts = sorted(set(A) & set(B))
    return ts, np.array([A[t] - B[t] for t in ts], dtype=float)


# --------------------------------------------------------------------------
# Terminology (principle 2)
# --------------------------------------------------------------------------
LABEL = {
    "gs": "GS",
    "gd": "GD",
    "gs_intensity": "GS (intensity target)",
    "gd_intensity": "GD (intensity target)",
    "transformer_per_target": "Transformer",
    # "shared" was ambiguous; this mirrors "Batched SAIL" so the two
    # batched arms are named the same way.
    "transformer_batched": "Batched Transformer",
    "sail": "SAIL",
    "batched_sail_750": "Batched SAIL (750)",
    "batched_sail_2000": "Batched SAIL (2000)",
    # Figures say "sim-seeded" (decision 2026-08-05: shorter, consistent
    # everywhere); manuscript PROSE spells out "simulation-seeded" at first
    # use, matching the GD+CITL paper. Never "warm start" anywhere. The DATA
    # keys keep the historical *_warm spelling; only display text changes.
    "gs_citl_random": "GS+CITL (random init)",
    "gs_citl_warm": "GS+CITL (sim-seeded)",
    "gd_citl_random": "GD+CITL (random init)",
    "gd_citl_warm": "GD+CITL (sim-seeded)",
    "fno_scratch": "FNO (from scratch)",
    "fno_regress": "FNO (regression onto GD phase)",
    # bench_replay-only methods (the converged baselines and the dropped
    # corrector, kept for the extended data):
    "gs_10000": "GS (10,000 iterations)",
    "gd_10000": "GD (10,000 iterations)",
    "sail_plus": "SAIL+",
}
PHYSICS_LABEL = {"ideal": "Ideal model", "faithful": "Faithful model"}
METRIC_LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "nmse": "NMSE",
                "mse": "MSE", "diffraction_efficiency": "Diffraction efficiency"}


def label(method: str) -> str:
    return LABEL.get(method, method)


# --------------------------------------------------------------------------
# Statistics (principle 3)
# --------------------------------------------------------------------------
def bootstrap_median_ci(x, n_boot: int = 10_000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the median. Seeded, so every rebuild
    of the manuscript produces identical intervals."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return (float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def wilcoxon(a, b=None) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank; b=None tests deltas against zero.

    Paired because every condition is measured on the same 18 targets.
    Non-parametric because n=18 with skew does not support a t-test, and a
    referee will ask why a t-test was used if one is.
    """
    from scipy.stats import wilcoxon as _w
    a = np.asarray(a, float)
    stat, p = _w(a) if b is None else _w(a, np.asarray(b, float))
    return float(stat), float(p)


def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, p.size + 1) / p.size)
    cutoff = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    out = np.zeros(p.size, dtype=bool)
    out[order[:cutoff]] = True
    return out


def summarize(x) -> dict:
    """Mean, SD, median, IQR, bootstrap CI, n -- everything T1 needs at once.

    Mean and SD are present because R2.8 asked for them; the median and its
    interval are what the narrative quotes (R2.9 asked why a median appeared
    without a mean, so both travel together from here on).
    """
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_median_ci(x)
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "q1": float(np.percentile(x, 25)),
            "q3": float(np.percentile(x, 75)), "ci_lo": lo, "ci_hi": hi,
            "n": int(x.size)}


def delta_summary(domain: str, physics: str, a: str, b: str,
                  metric: str = "psnr", path=None) -> dict:
    """The standard paired comparison quoted throughout the record."""
    ts, d = paired(domain, physics, a, b, metric, path)
    s = summarize(d)
    s.update({"a": a, "b": b, "physics": physics, "metric": metric,
              "wins": int((d > 0).sum()), "n_pairs": len(ts),
              "min": float(d.min()) if d.size else float("nan"),
              "max": float(d.max()) if d.size else float("nan")})
    s["p"] = wilcoxon(d)[1] if d.size and np.any(d != 0) else float("nan")
    return s


# --------------------------------------------------------------------------
# Style -- OWNED BY figstyle.py, re-exported here so existing figure code that
# says `S.method_color(...)` keeps working. New figure modules should import
# figstyle directly. Nothing visual is defined in this file.
# --------------------------------------------------------------------------
from figstyle import (  # noqa: E402,F401
    FONT, SIZE, RC, font_available, apply_style,
    COOL, WARM, PURPLE, ORANGE, PINK, GREEN, ORANGE_F, PURPLE_F, BLUE,
    CANONICAL_ORDER, ACCENT, ACCENT_FAINT,
    COLOR_DIAG, COLOR_FAINT, COLOR_SAIL, COLOR_BATCHED, COLOR_TRANSFORMER,
    COLOR_GS, COLOR_GD, COLOR_CITL,
    METHOD_COLORS, method_color, is_ours,
    add_panel_label, style_axis, padded_limits, save, halo,
)
from figstyle import palette_table as _palette_table


def palette_table() -> None:
    """Print method -> colour with the paper's display names."""
    _palette_table(label_fn=label)


# --------------------------------------------------------------------------
def check(path=None) -> None:
    """Inventory the canonical dataset. Run before building anything."""
    d = load(path)
    print(f"scored dataset: {d['_path']}")
    print(f"  written: {d['meta']['written']}   records: {len(d['records'])}")
    seen = {}
    for r in d["records"]:
        seen.setdefault((r["domain"], r["physics"]), set()).add(r["method"])
    for k in sorted(seen):
        n = {m: len(records(k[0], k[1], m, path)) for m in sorted(seen[k])}
        short = {m: c for m, c in n.items() if c != N_TARGETS}
        print(f"  {k[0]:10s} {k[1]:8s} {len(seen[k]):2d} methods"
              + (f"   INCOMPLETE: {short}" if short else "   all 18/18"))
    ts = targets(path)
    print(f"  targets: {len(ts)}"
          + ("" if len(ts) == N_TARGETS else f"  <- expected {N_TARGETS}"))
    tim = [t for t in d["timings"] if t["seconds"] is not None]
    print(f"  timings: {len(tim)}/{len(d['timings'])} rows with wall-clock")


# --------------------------------------------------------------------------
# Saved summary tables
#
# Every build that prints summary statistics ALSO saves them, so a quoted
# number is always recoverable from a file the notebook wrote rather than
# from a scrolled-away cell print. One helper, two formats: CSV (full
# precision, machine-readable) and Markdown (readable in place). Written to
# the same analysis tables/ directory as T1-T4.
# --------------------------------------------------------------------------
def _fmt(v, precision: int = 6) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.{precision}g}"
    return str(v)


def write_table(table_dir, stem: str, header: list[str], rows: list[list],
                note: str = "") -> Path:
    """Save a summary table as <stem>.csv and <stem>.md in table_dir.

    header: column names. rows: lists of values (numbers or strings); floats
    are written at full working precision in the CSV and 6 significant
    figures in the Markdown. note: provenance line appended to the Markdown
    and as a '# ' comment line at the top of the CSV.
    """
    import csv

    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        if note:
            f.write(f"# {note}\n")
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v, precision=12) for v in r])

    md_path = table_dir / f"{stem}.md"
    cells = [[_fmt(v) for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(header)]
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
              for r in cells]
    if note:
        lines += ["", note]
    md_path.write_text("\n".join(lines) + "\n")

    print(f"  table -> {csv_path}")
    print(f"  table -> {md_path}")
    return csv_path


def summary_row(s: dict) -> list:
    """The standard column set for a summarize() dict, matching
    SUMMARY_HEADER. Keeps every saved table's statistics columns identical."""
    return [s["median"], s["mean"], s["sd"], s["q1"], s["q3"],
            s["ci_lo"], s["ci_hi"], s["n"]]


SUMMARY_HEADER = ["median", "mean", "sd", "q1", "q3",
                  "ci_lo", "ci_hi", "n"]
