r"""
replay_analysis.py -- what the 2026-08-04 single-alignment re-capture says.

Reads sail_scored.json and NOTHING else. Two reports, two different
questions:

report_agreement()
    July versus August on the methods both capture sets share. The submitted
    manuscript's bench numbers come from July-era captures (domain "bench");
    the revision's come from the 2026-08-04 replay (domain "bench_replay").
    Where the SAME displayed phase was photographed on both dates, the PSNR
    difference measures rig reproducibility across eleven-plus days and
    nothing else, and it is the validation gate for
    switching the manuscript's bench figures from "bench" to "bench_replay".

    Pairs are classified honestly:
      same_phase : the displayed hologram is the identical stored array.
                   Verified for transformer_per_target 2026-08-04: the July
                   sweep's phase and the 10k sweep's phase are bit-identical
                   (sha256, alley/ideal), so the forward pass is deterministic.
                   sail/batched_sail/CITL replay the stored best phase that the
                   July "live" capture also displayed.
      reseeded   : gs_750/gd_750 replay the 10k sweep's OWN runs, which are
                   fresh random initialisations, not the July sweep's arrays.
                   Their deltas fold optimiser reseeding into the rig number,
                   so they are reported separately and NOT counted in the
                   reproducibility headline.

report_convergence()
    The full bench table: ALL 14 captured methods per physics, every row from
    domain "bench_replay" so every number shares one alignment. Headlined by
    the question the convergence thread comes down to: GD gains +17.75 dB from
    750 to 10,000 iterations in ideal simulation; the gd_10000 - gd_750 delta
    ON THE BENCH says how much of that survives the optics. That is the number
    for the open bracket in the R2 draft. sail_plus is in the table because it
    was captured, and stays out of the manuscript regardless.

CAVEAT carried from the capture manifest: transformer_batched under the
faithful condition ran hot on the saturation check (mean 0.72% of ROI pixels
at or above 250, max 2.05%, roughly 6x every other method). Its faithful
values may be compressed at the bright end; any close comparison involving
that one cell needs a caveat, and both reports print one when it appears.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# (bench method, bench_replay method, same displayed phase?)
PAIRS = [
    ("gs",                     "gs_750",                 False),
    ("gd",                     "gd_750",                 False),
    ("transformer_per_target", "transformer_per_target", True),
    ("transformer_batched",    "transformer_batched",    True),
    ("sail",                   "sail",                   True),
    ("batched_sail_750",       "batched_sail_750",       True),
    ("batched_sail_2000",      "batched_sail_2000",      True),
    ("gs_citl_random",         "gs_citl_random",         True),
    ("gs_citl_warm",           "gs_citl_warm",           True),
    ("gd_citl_random",         "gd_citl_random",         True),
    ("gd_citl_warm",           "gd_citl_warm",           True),
]

SATURATION_CAVEAT = ("transformer_batched", "faithful")


def _load(scored_path=None) -> dict:
    p = Path(scored_path) if scored_path else Path(os.environ["SAILREV_OUT"])
    d = json.loads(p.read_text())
    idx: dict[tuple, dict] = {}
    for r in d["records"]:
        idx[(r["domain"], r["physics"], r["method"], r["target"])] = r
    return idx


def _targets(idx, domain, physics, method):
    return sorted(t for (d, p, m, t) in idx
                  if d == domain and p == physics and m == method)


def report_agreement(scored_path=None) -> dict:
    """July (bench) vs 2026-08-04 (bench_replay), paired by target name."""
    idx = _load(scored_path)
    out = {}
    print("Replay agreement: bench (July) vs bench_replay (2026-08-04)")
    print("delta = replay - July, dB. Pairing by target name, never position.\n")
    headline = []
    for physics in ("ideal", "faithful"):
        print(f"[{physics}]")
        print(f"  {'method':<26}{'n':>3}{'mean d':>9}{'median d':>10}"
              f"{'max |d|':>9}   phase")
        for old_m, new_m, same in PAIRS:
            ts = [t for t in _targets(idx, "bench", physics, old_m)
                  if ("bench_replay", physics, new_m, t) in idx]
            if not ts:
                continue
            d = np.array([idx[("bench_replay", physics, new_m, t)]["psnr"]
                          - idx[("bench", physics, old_m, t)]["psnr"]
                          for t in ts])
            tag = "same" if same else "RESEEDED"
            note = ""
            if (new_m, physics) == SATURATION_CAVEAT:
                note = "  <-- saturation caveat: faithful captures ran hot"
            print(f"  {new_m:<26}{len(ts):>3}{d.mean():>+9.2f}"
                  f"{np.median(d):>+10.2f}{np.abs(d).max():>9.2f}   {tag}{note}")
            out[(physics, new_m)] = d
            if same and (new_m, physics) != SATURATION_CAVEAT:
                headline.append(d)
        print()
    if headline:
        h = np.concatenate(headline)
        print(f"RIG REPRODUCIBILITY (same-phase pairs only, saturation-caveat "
              f"cell excluded, n={len(h)}):")
        print(f"  mean delta {h.mean():+.2f} dB, median |delta| "
              f"{np.median(np.abs(h)):.2f} dB, 95th pct |delta| "
              f"{np.percentile(np.abs(h), 95):.2f} dB, max |delta| "
              f"{np.abs(h).max():.2f} dB")
        print("  The same holograms, re-photographed 11 days later, reproduce")
        print("  to within the figures above. This is the validation gate for")
        print("  quoting bench_replay in the manuscript.")
    return out


def report_convergence(scored_path=None) -> dict:
    """Converged GD/GS on the bench, against the sweep's simulation numbers."""
    idx = _load(scored_path)
    # GD 750->10000, amplitude, MEDIAN across 18 targets, matching E5 and
    # Fig 2. The superseded mean values were 17.75/0.63.
    sim_gain = {"ideal": 19.09, "faithful": 0.65}
    out = {}
    print("Converged baselines on the bench (all rows from bench_replay: one")
    print("session, one alignment, 2026-08-04)\n")
    for physics in ("ideal", "faithful"):
        rows = []
        for m in ("gs_750", "gs_10000", "gd_750", "gd_10000",
                  "transformer_per_target", "transformer_batched",
                  "sail", "sail_plus",
                  "batched_sail_750", "batched_sail_2000",
                  "gs_citl_random", "gs_citl_warm",
                  "gd_citl_random", "gd_citl_warm"):
            ts = _targets(idx, "bench_replay", physics, m)
            if not ts:
                continue
            v = np.array([idx[("bench_replay", physics, m, t)]["psnr"]
                          for t in ts])
            rows.append((m, len(ts), v.mean(), np.median(v), v.std(ddof=1)))
            out[(physics, m)] = v
        if not rows:
            continue
        print(f"[{physics}]")
        print(f"  {'method':<26}{'n':>3}{'mean':>8}{'median':>8}{'sd':>7}")
        for m, n, mu, med, sd in rows:
            note = ("  <-- saturation caveat"
                    if (m, physics) == SATURATION_CAVEAT else "")
            print(f"  {m:<26}{n:>3}{mu:>8.2f}{med:>8.2f}{sd:>7.2f}{note}")
        k750, k10k = (physics, "gd_750"), (physics, "gd_10000")
        if k750 in out and k10k in out:
            ts_a = _targets(idx, "bench_replay", physics, "gd_750")
            ts_b = _targets(idx, "bench_replay", physics, "gd_10000")
            common = sorted(set(ts_a) & set(ts_b))
            d = np.array([idx[("bench_replay", physics, "gd_10000", t)]["psnr"]
                          - idx[("bench_replay", physics, "gd_750", t)]["psnr"]
                          for t in common])
            print(f"\n  GD 750 -> 10000 ON THE BENCH: "
                  f"{np.median(d):+.2f} dB median "
                  f"(mean {d.mean():+.2f}, n={len(d)})")
            print(f"  the same step in {physics} SIMULATION:  "
                  f"{sim_gain[physics]:+.2f} dB median")
            if physics == "ideal":
                print(f"  -> {np.median(d) / sim_gain[physics] * 100:.0f}% "
                      f"of the simulation gain survives the optics")
        print()
    print("Fill the R2 draft's open bracket from the ideal block above.")
    return out
