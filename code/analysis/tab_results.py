r"""
tab_results.py -- T1 (full results) and T2 (compute and cost).

T1 answers two reviewer points at once and must keep doing so:
  R2.8 "Average PSNR and SSIM values (not only plots), and standard deviations
       should be reported for all methods"      -> mean and SD, every method.
  R2.9 "Why is the median reported instead of the more commonly used average?"
       -> median AND mean travel together everywhere, so the question cannot
          recur. The narrative quotes the median (skewed, n=18); the table
          shows both and the reader can see they agree in direction.

T2 is the amortisation table. It exists because the batched-SAIL claim is
easy to overstate and a reviewer will check it:

  Batched training is NOT faster than per-target training. Both perform one
  camera exposure per target per epoch, so 750 epochs x 18 targets = 13,500
  exposures either way, and the camera dominates wall-clock. Measured: batched
  750 costs 8.35 hr (ideal) against 6.82 hr for the 18 per-target runs summed.

  What batched training buys is a SINGLE model that serves any target in one
  forward pass. The honest cost metric is therefore the MARGINAL cost of a new
  target once training is done: one forward pass for a learned model, versus a
  complete camera-in-the-loop optimisation for GD+CITL (median 21.3 min,
  ideal). That is the number this table leads with.

Every wall-clock figure is read from the run logs via sail_scored.json's
timings table. Values that cannot be derived from the dataset are declared in
CONSTANTS below with their provenance, so nothing enters a table by assertion
without a stated source.

Usage (notebook):
    import tab_results
    tab_results.build(OUT / "tables")
"""
from __future__ import annotations

import collections
import statistics as st
from pathlib import Path

import numpy as np

import sailrev as S

# --------------------------------------------------------------------------
# CONSTANTS -- not derivable from sail_scored.json. Each carries its
# source. VERIFY these against run_configuration.json before submission; they
# are the only numbers in this module that are asserted rather than measured.
# --------------------------------------------------------------------------
CONSTANTS = {
    "hot_params": (195_660_320, "attention_ablation.py control arm; p=500, "
                                "d_model=256, 16 heads, 4 layers"),
    "fno_params": (256_001_730, "fno_rebuttal_results.json -> per_target "
                                "conditions.fno_scratch.params"),
    "attention_params": (1_052_672, "self-attention sublayers only "
                                    "(hot minus hot_no_attn)"),
    "sim_epochs": (10_000, "Methods 5: per-target simulation training"),
    "sail_epochs": (750, "sail_citl_transformer.py run configuration"),
    "gd_iterations": (750, "reporting iteration, matched sim and bench"),
    "gs_iterations_bench": (25, "camera-feedback GS budget"),
    "inference_seconds": (0.015, "Ext Fig 1D, single forward pass; RE-MEASURE "
                                 "on the current machine before quoting"),
}

METRICS = ("psnr", "ssim", "nmse")
# The 10,000-iteration bench arms are IN T1 (2026-08-09): the manuscript
# claims every simulation-only method lands in the same band on hardware
# even when GD is given 13x the compute, and that claim needs a row a
# reader can check. Fig 3 stays at the 750 operating point.
BENCH_ORDER = ["gs", "gs_10000", "gd", "gd_10000",
               "gs_citl_random", "gs_citl_warm", "gd_citl_random",
               "gd_citl_warm", "transformer_per_target", "transformer_batched",
               "sail", "batched_sail_750", "batched_sail_2000"]
SIM_ORDER = ["gs", "gs_10000", "gs_intensity",
             "gd", "gd_10000", "gd_intensity",
             "transformer_per_target", "transformer_batched",
             "fno_scratch", "fno_regress"]


# --------------------------------------------------------------------------
# T1
# --------------------------------------------------------------------------
def t1_rows(domain: str, path=None) -> list[dict]:
    order = BENCH_ORDER if domain == "bench" else SIM_ORDER
    rows = []
    for physics in S.PHYSICS:
        present = {r["method"] for r in S.records(domain, physics, path=path)}
        for m in order:
            if m not in present:
                continue
            row = {"domain": domain, "physics": physics, "method": m,
                   "label": S.label(m)}
            for metric in METRICS:
                v = list(S.by_target(domain, physics, m, metric, path).values())
                s = S.summarize(v)
                row[metric] = s
            rows.append(row)
    return rows


def print_t1(domain: str, path=None) -> None:
    print(f"\nT1 | {domain}  (n=18 targets; mean +/- SD, and median [IQR])")
    print(f"{'physics':9s} {'method':28s} {'PSNR mean+/-SD':>18s} "
          f"{'PSNR median [IQR]':>24s} {'SSIM mean':>10s} {'NMSE mean':>10s}")
    for r in t1_rows(domain, path):
        p, ss, nm = r["psnr"], r["ssim"], r["nmse"]
        print(f"{r['physics']:9s} {r['label']:28s} "
              f"{p['mean']:9.2f} +/- {p['sd']:5.2f} "
              f"{p['median']:9.2f} [{p['q1']:.2f}, {p['q3']:.2f}] "
              f"{ss['mean']:10.3f} {nm['mean']:10.4f}")


def t1_tex(domain: str, out_dir: Path, path=None) -> Path:
    rows = t1_rows(domain, path)
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Model & Method & PSNR (dB) & PSNR median & SSIM & NMSE \\",
         r" & & mean $\pm$ SD & [IQR] & mean $\pm$ SD & mean $\pm$ SD \\",
         r"\midrule"]
    last = None
    for r in rows:
        phys = S.PHYSICS_LABEL[r["physics"]] if r["physics"] != last else ""
        last = r["physics"]
        if phys and L[-1] != r"\midrule":
            L.append(r"\midrule")
        p, ss, nm = r["psnr"], r["ssim"], r["nmse"]
        L.append(f"{phys} & {r['label']} & "
                 f"${p['mean']:.2f} \\pm {p['sd']:.2f}$ & "
                 f"${p['median']:.2f}$ [{p['q1']:.2f}, {p['q3']:.2f}] & "
                 f"${ss['mean']:.3f} \\pm {ss['sd']:.3f}$ & "
                 f"${nm['mean']:.4f} \\pm {nm['sd']:.4f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    out = Path(out_dir) / f"t1_results_{domain}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"  table -> {out}")
    return out


# --------------------------------------------------------------------------
# T2
# --------------------------------------------------------------------------
def timings_summary(path=None) -> dict:
    """{(method, physics): {'n', 'total_hr', 'median_min'}} from the run logs."""
    g = collections.defaultdict(list)
    for t in S.load(path)["timings"]:
        if t["seconds"] is not None:
            g[(t["method"], t["physics"])].append(t["seconds"])
    return {k: {"n": len(v), "total_hr": sum(v) / 3600.0,
                "median_min": st.median(v) / 60.0} for k, v in g.items()}


def t2_rows(physics: str = "ideal", path=None) -> list[dict]:
    """Cost of the eighteen-target campaign, and of one further target."""
    T = timings_summary(path)
    inf = CONSTANTS["inference_seconds"][0]
    rows = []

    def add(label, models, train_hr, marginal, note):
        rows.append({"label": label, "models": models, "train_hr": train_hr,
                     "marginal": marginal, "note": note})

    gd = T.get(("gd_citl_random", physics))
    if gd:
        add(S.label("gd_citl_random"), "none (no network)", gd["total_hr"],
            f"{gd['median_min']:.1f} min", "full optimisation per new target")
    gdw = T.get(("gd_citl_warm", physics))
    if gdw:
        add(S.label("gd_citl_warm"), "none (no network)", gdw["total_hr"],
            f"{gdw['median_min']:.1f} min",
            "still a full optimisation; seeded from simulation")
    sail = T.get(("sail", physics))
    if sail:
        add(S.label("sail") + " (per target)", "18 networks", sail["total_hr"],
            f"{sail['median_min']:.1f} min",
            "a new target needs its own network and its own camera session")
    for k, lab in (("batched_sail_750", "Batched SAIL (750 epochs)"),
                   ("batched_sail_2000", "Batched SAIL (2000 epochs)")):
        b = T.get((k, physics))
        if b:
            add(lab, "1 shared network", b["total_hr"], f"{inf:.3f} s",
                "one forward pass; no camera, no optimisation")
    return rows


def print_t2(physics: str = "ideal", path=None) -> None:
    rows = t2_rows(physics, path)
    print(f"\nT2 | cost of the 18-target campaign and of one further target "
          f"({S.PHYSICS_LABEL[physics]})")
    print(f"{'method':30s} {'networks':16s} {'train (hr)':>11s} "
          f"{'per new target':>15s}   note")
    for r in rows:
        print(f"{r['label']:30s} {r['models']:16s} {r['train_hr']:11.2f} "
              f"{r['marginal']:>15s}   {r['note']}")
    print("\nRead this table for the LAST column, not the third. Batched "
          "training is not\ncheaper than per-target training (same number of "
          "camera exposures, camera-bound);\nit yields one network instead of "
          "eighteen, and that is what collapses the cost of\nevery subsequent "
          "target. GS and GD have no weights to share, so the shared-network\n"
          "row has no classical counterpart at all.")


def t2_tex(physics: str, out_dir: Path, path=None) -> Path:
    rows = t2_rows(physics, path)
    L = [r"\begin{tabular}{llrr}", r"\toprule",
         r"Method & Networks produced & Campaign (hr) & Per target served \\",
         r"\midrule"]
    for r in rows:
        L.append(f"{r['label']} & {r['models']} & {r['train_hr']:.2f} & "
                 f"{r['marginal']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    out = Path(out_dir) / f"t2_compute_{physics}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"  table -> {out}")
    return out


# --------------------------------------------------------------------------
def build(out_dir, path=None) -> None:
    out_dir = Path(out_dir)
    for domain in ("simulation", "bench"):
        print_t1(domain, path)
        t1_tex(domain, out_dir, path)
    for physics in S.PHYSICS:
        print_t2(physics, path)
        t2_tex(physics, out_dir, path)
    print("\nAsserted constants (verify against run_configuration.json):")
    for k, (v, src) in CONSTANTS.items():
        print(f"  {k:22s} {v!s:>12s}   {src}")
