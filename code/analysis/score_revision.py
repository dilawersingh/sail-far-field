r"""
score_revision.py -- THE canonical scoring pass. Every number quoted in
the manuscript comes from the JSON this script writes, and from nowhere
else.

One record per (domain, physics, method, target), all scored by
`evaluate_methods.compute_metrics` -- the project's single scoring function,
imported, never reimplemented (same discipline as the GD+CITL paper's
score_captures.py, which this generalises).

DOMAINS AND METHODS
    simulation : gs, gd, gs_intensity, gd_intensity   (iter 750 recon .npy,
                 read from the matched-compute 10k SWEEP -- decision
                 2026-08-05: one GS/GD dataset everywhere, 750 read off the
                 sweep, the same runs whose phases the bench replay
                 photographed. See sim_comparison_dirs().)
                 transformer_per_target                (transformer_recon.npy)
                 transformer_batched                   (recomputed from
                                                        {stem}_best_phase.npy
                                                        through physics.py)
                 fno_scratch, fno_regress              (ideal only, by design)
    bench      : gs, gd, transformer_per_target, transformer_batched
                                                       (replay_simulation run)
                 sail                                  (per-target SAIL)
                 batched_sail_750, batched_sail_2000   (shared model)
                 gs_citl_random, gs_citl_warm          (camera-feedback GS)
                 gd_citl_random, gd_citl_warm          (camera-feedback GD)
    bench_replay : the 2026-08-04 full re-capture (replay_converged run):
                 EVERY method above photographed again in ONE session on ONE
                 alignment, plus gs_10000/gd_10000 (converged baselines),
                 gs_750/gd_750 (operating point, re-drawn from the 10k sweep's
                 own random inits) and sail_plus. Methods are DISCOVERED from
                 the capture files rather than listed here, so a future replay
                 with more methods scores without a code change.

    bench and bench_replay are BOTH kept. bench is the July-era evidence behind
    the submitted manuscript; bench_replay is the single-alignment dataset the
    revision quotes. Their overlap on same-displayed-phase methods measures rig
    reproducibility across the two dates (see replay_analysis.py). Verified
    2026-08-04: the per-target transformer phase written by the July sweep and
    by the 10k sweep is bit-identical (sha256, alley/ideal), so the forward
    pass is deterministic and those pairs isolate the rig, not the model.

PHYSICS: ideal and faithful wherever both exist. FNO is ideal-only by decision
(2026-08-03): its from-scratch quality is too far below every other method for
the faithful re-run to inform anything.

RESOLUTION RULE (from score_captures.py, applied uniformly): every comparison
is scored with target and reconstruction on the SAME grid. Captures are loaded
through the shared calibration at (H*pad_factor, W*pad_factor) with the DC
exclusion scaled by pad_factor; faithful simulation recons arrive at 2000x2000
already. compute_metrics receives pad_factor so the diffraction-efficiency
support threshold stays resolution-invariant.

CAPTURE RULE: wherever a raw DSLR capture exists it is re-processed HERE
through the one shared loader and the one rig calibration, so no asymmetry can
enter at scoring time. Stored processed captures are used only as fallback and
the record says so (capture_kind).

WALL-CLOCK: a second table, `timings`, collects wall-clock per run where the
logs record it -- per-target SAIL, batched SAIL (the headline: ONE camera
session trains all 18 targets; a single forward pass then serves any of them),
GD+CITL (gdcitl.run_metadata), and the simulation sweeps' wall_clock_seconds.

Usage
    python score_revision.py --dry-run     # inventory only, score nothing
    python score_revision.py               # writes sail_scored.json
    python score_revision.py --only bench_sail,sim_gd   # subset while testing
    python score_revision.py --only replay_gd_10000     # bench_replay methods

Immutability: reads the results trees, writes ONLY to --out (default
<Results>/Self-Attention/multilevel/analysis/sail_scored.json -- the
revision's own analysis folder, kept apart from Transformers/nature_publication,
which holds the first submission's results).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Path resolution -- ONE root, set in ONE place.
#
# This module carries NO paths of its own. The analysis notebook's Cell 0 (or
# the shell, for standalone/deposit runs) sets two environment variables
# BEFORE this module is imported:
#
#   SAILREV_RESULTS   the Results root      (required)
#   SAILREV_REPO      the holography repo   (required)
#
# Everything else derives from SAILREV_RESULTS by the tree's own fixed
# structure. The deposit re-points the entire pipeline by setting these two
# variables to its own root -- same single-root convention as the GD+CITL
# deposit. Optional overrides exist only for the case where the deposit
# splits a sub-tree out: SAILREV_SA, SAILREV_GDCITL, SAILREV_FNO,
# SAILREV_STOCK, SAILREV_OUT.
# --------------------------------------------------------------------------
import os


def _required_env(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set. Set it in the analysis notebook's Cell 0 "
            f"(or the shell) before importing score_revision -- this module "
            f"deliberately carries no paths of its own."
        )
    return Path(v)


def _env(name: str, default: Path) -> Path:
    return Path(os.environ[name]) if os.environ.get(name) else default


RESULTS = _required_env("SAILREV_RESULTS")
REPO = _required_env("SAILREV_REPO")
SA_ROOT = _env("SAILREV_SA", RESULTS / "Self-Attention" / "multilevel")
GDCITL_ROOT = _env("SAILREV_GDCITL", RESULTS / "GD+CITL")
FNO_ROOT = _env("SAILREV_FNO", RESULTS / "FNO")
STOCK = _env("SAILREV_STOCK", RESULTS / "Stock Images" / "1000x1000")
# Revision outputs live in the Self-Attention tree, deliberately apart from
# Transformers/nature_publication (the first submission's results):
OUT_DEFAULT = _env("SAILREV_OUT", SA_ROOT / "analysis" / "sail_scored.json")

FORWARD_MODELS = {
    "ideal":    dict(pad_factor=1, apply_sinc=False, fill_factor=1.0, fill_is_areal=True),
    "faithful": dict(pad_factor=2, apply_sinc=True, fill_factor=0.91, fill_is_areal=True),
}
DC_CENTER_NATIVE = (495, 510)   # from run_configuration.json -> hardware
DC_RADIUS_NATIVE = 20
GD_ITER = 750                   # reporting iteration, matched sim/bench
N_TARGETS = 18


# --------------------------------------------------------------------------
# Discovery helpers. Each returns {target: <path or dict>} and never raises on
# absence -- missing entries surface in the inventory instead.
# --------------------------------------------------------------------------
def _newest(dirs):
    dirs = sorted(dirs)
    return dirs[-1] if dirs else None


def _newest_per_target(parent: Path, suffix_re: str) -> dict[str, Path]:
    """Map target -> newest '<target>_<...>' run dir under parent."""
    out: dict[str, Path] = {}
    if not parent.exists():
        return out
    rx = re.compile(suffix_re)
    for d in sorted(p for p in parent.iterdir() if p.is_dir()):
        m = rx.match(d.name)
        if m:
            out[m.group(1)] = d           # sorted() -> newest timestamp wins
    return out


def sim_comparison_dirs(physics: str) -> dict[str, Path]:
    # DECISION 2026-08-05 (Dilawer): the simulation domain scores the
    # matched-compute 10k sweep, NOT the pre-sweep comparison in the GD+CITL
    # tree. Reason: the sweep's 750- and 10,000-iteration phases are the ones
    # the bench replay photographed (simulation_capture_converged.json resolves
    # gs_750/gd_750/gd_10000/transformer from the sweep tree), so scoring the
    # sweep makes the simulation number and the bench number refer to the SAME
    # hologram, and leaves exactly one GS/GD simulation dataset with 750 read
    # off it. The pre-sweep comparison stays untouched in the GD+CITL tree (it
    # belongs to that paper) and is used here ONLY by score_original_sim_gd()
    # to verify the scorer against the GD+CITL pipeline's scored.json.
    base = SA_ROOT / "simulations" / f"simulation_comparison_{physics}_10k"
    return _newest_per_target(base, r"(.+?)_simulation_comparison.*")


def sim_comparison_dirs_original(physics: str) -> dict[str, Path]:
    """The pre-sweep comparison run in the GD+CITL tree. VERIFICATION-ONLY:
    its scores never enter sail_scored.json; the notebook's cross-check
    cell scores these files on the fly and compares them against the GD+CITL
    pipeline's scored.json, which was produced from these same files."""
    base = GDCITL_ROOT / "simulations" / f"simulation_comparison_{physics}"
    return _newest_per_target(base, r"(.+?)_simulation_comparison.*")


def sim_recon(target_dir: Path, method: str) -> Path | None:
    """method: gs | gd | gs_10000 | gd_10000 | gs_intensity | gd_intensity
    | transformer_per_target

    The converged arms cost nothing to score. The sweep wrote a recon at
    every budget it passed through, including 10,000, in the SAME tree the
    750-iteration operating point is read from. Scoring them puts the
    converged simulation numbers in T1 beside the converged bench numbers,
    so both halves are present: GD reaches 56 dB in simulation at 10,000
    iterations and still lands at 16 dB on the bench.
    """
    if target_dir is None:
        return None
    if method == "transformer_per_target":
        p = target_dir / "outputs" / "transformer" / "transformer_recon.npy"
        return p if p.exists() else None
    sub = {"gs": "gs", "gd": "gd",
           "gs_10000": "gs", "gd_10000": "gd",
           "gs_intensity": "gs_intensity_target",
           "gd_intensity": "gd_intensity_target"}[method]
    stem = method.split("_")[0]
    iters = 10000 if method.endswith("_10000") else GD_ITER
    # Intensity-formulation sweeps tag their filenames: gs_iter_750_intensity_recon.npy
    fname = (f"{stem}_iter_{iters}_intensity_recon.npy"
             if method.endswith("_intensity")
             else f"{stem}_iter_{iters}_recon.npy")
    for rel in (Path("outputs") / sub, Path(sub)):
        p = target_dir / rel / fname
        if p.exists():
            return p
    return None


def batched_sim_phase(physics: str) -> dict[str, Path]:
    base = SA_ROOT / "simulations" / "batched"
    run = _newest(p for p in base.iterdir()
                  if p.is_dir() and f"_{physics}_" in p.name) if base.exists() else None
    if run is None:
        return {}
    out = {}
    for p in (run / "outputs").glob("*_best_phase.npy"):
        out[p.name[:-len("_best_phase.npy")]] = p
    return out


def fno_recons() -> dict[str, dict[str, Path]]:
    base = FNO_ROOT / "rebuttal"
    run = _newest(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("fno_rebuttal_")) if base.exists() else None
    if run is None:
        return {}
    out: dict[str, dict[str, Path]] = {}
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        rec = {}
        for cond in ("fno_scratch", "fno_regress"):
            p = d / f"{cond}_recon.npy"
            if p.exists():
                rec[cond] = p
        if rec:
            out[d.name] = rec
    return out


def replay_captures(physics: str) -> dict[str, dict[str, Path]]:
    base = SA_ROOT / "experiments" / "replay_simulation"
    run = _newest(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("replay_simulation_")) if base.exists() else None
    if run is None:
        return {}
    out: dict[str, dict[str, Path]] = {}
    cond = run / physics
    if not cond.exists():
        return {}
    for d in sorted(p for p in cond.iterdir() if p.is_dir()):
        rec = {m: d / f"{m}.jpg"
               for m in ("gs", "gd", "transformer_per_target", "transformer_batched")
               if (d / f"{m}.jpg").exists()}
        if rec:
            out[d.name] = rec
    return out


def replay_converged_captures(physics: str) -> dict[str, dict[str, Path]]:
    """target -> {method: capture path} from the newest replay_converged run.

    Method names are discovered from the .jpg files present (same rule as
    build_dataset_manifest after 2026-08-04): every {target}/{key}.jpg becomes
    method {key}, names starting with '_' ignored. Discovery rather than a
    list, so the scorer cannot silently drop a method the capture run added.
    """
    base = SA_ROOT / "experiments" / "replay_converged"
    run = _newest(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("replay_converged_")) if base.exists() else None
    if run is None:
        return {}
    cond = run / physics
    if not cond.exists():
        return {}
    out: dict[str, dict[str, Path]] = {}
    for d in sorted(p for p in cond.iterdir() if p.is_dir()):
        rec = {p.stem: p for p in sorted(d.glob("*.jpg"))
               if not p.stem.startswith("_")}
        if rec:
            out[d.name] = rec
    return out


_BEST_EPOCH_RX = (re.compile(r"[Bb]est\s+(?:epoch|iteration)[:\s]+(\d+)"),
                  re.compile(r"epoch\s+(\d+).*?\bbest\b", re.I))


def _best_epoch_from_log(log: Path) -> int | None:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="replace")
    for rx in _BEST_EPOCH_RX:
        hits = rx.findall(txt)
        if hits:
            return int(hits[-1])
    return None


def _sail_capture(run_dir: Path) -> dict:
    """Raw best-epoch capture if identifiable, else stored processed best."""
    raw_dir = run_dir / "dslr captures"
    best = _best_epoch_from_log(run_dir / "experimental_log.txt")
    if best is None and raw_dir.exists():
        eps = sorted(int(m.group(1)) for p in raw_dir.glob("epoch_*.jpg")
                     if (m := re.match(r"epoch_(\d+)\.jpg", p.name)))
        if len(eps) == 3:            # first / best / last retained
            best = eps[1]
    if best is not None:
        p = raw_dir / f"epoch_{best:04d}.jpg"
        if p.exists():
            return {"capture": p, "kind": "raw", "best": best}
    p = run_dir / "outputs" / "best_camera_capture.png"
    return {"capture": p if p.exists() else None, "kind": "processed", "best": best}


def sail_runs(physics: str) -> dict[str, dict]:
    base = SA_ROOT / "experiments" / "sail" / physics
    dirs = _newest_per_target(base, rf"(.+?)_{physics}_\d{{8}}_\d{{6}}$")
    return {t: {"run_dir": d, **_sail_capture(d)} for t, d in dirs.items()}


def batched_sail_runs(root_name: str, physics: str) -> dict[str, dict]:
    base = SA_ROOT / "experiments" / root_name / physics
    run = _newest(p for p in base.iterdir() if p.is_dir()) if base.exists() else None
    if run is None:
        return {}
    out = {}
    for d in sorted(p for p in (run / "targets").iterdir() if p.is_dir()):
        out[d.name] = {"run_dir": run, **_sail_capture(d)}
    return out


def _best_from_convergence_csv(csv_path: Path) -> int | None:
    """Best iteration = argmin of the loss column in a convergence CSV.

    Needed for GS: its camera-feedback runs do NOT log a 'best iteration' line
    (only GD does), but every run writes gs_camera_feedback_convergence.csv.
    Parsed defensively: first column is taken as iteration, the last numeric
    column as loss.
    """
    if not csv_path.exists():
        return None
    import csv as _csv
    best_it, best_loss = None, None
    with csv_path.open(newline="") as f:
        for row in _csv.reader(f):
            if not row:
                continue
            try:
                it = int(float(row[0]))
                loss = float(row[-1])
            except (ValueError, IndexError):
                continue        # header or malformed line
            if best_loss is None or loss < best_loss:
                best_it, best_loss = it, loss
    return best_it


def citl_runs(init: str, physics: str, alg: str) -> dict[str, dict]:
    """alg: gd | gs. Best-iteration capture of the camera-feedback baseline.

    GD's best iteration comes from the run log ('GD: best iteration N'); GS has
    no such line, so it comes from gs_camera_feedback_convergence.csv. Either
    way the retained captures are first/best/last, so the best capture exists.
    """
    base = GDCITL_ROOT / "experiments" / init / f"citl_gs_gd_{physics}"
    out: dict[str, dict] = {}
    if not base.exists():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        log = d / "experimental_log.txt"
        if not log.exists():
            continue
        txt = log.read_text(encoding="utf-8", errors="replace")
        m_t = re.search(r"Target:\s*(\S+)", txt)
        if not m_t:
            continue
        m_b = re.search(alg.upper() + r":\s*best iteration\s*(\d+)", txt)
        best = (int(m_b.group(1)) if m_b else
                _best_from_convergence_csv(d / f"{alg}_camera_feedback_convergence.csv"))
        if best is None:
            continue
        raw = d / "dslr captures" / f"{alg}_iter_{best:04d}.jpg"
        proc = d / "processed" / f"{alg}_captures" / f"{alg}_iter_{best:04d}.png"
        cap, kind = (raw, "raw") if raw.exists() else \
                    (proc, "processed") if proc.exists() else (None, None)
        out[m_t.group(1)] = {"run_dir": d, "capture": cap, "kind": kind,
                             "best": best}
    return out


def target_image(target: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = STOCK / f"{target}{ext}"
        if p.exists():
            return p
    return None


def calibration() -> dict:
    # One rig calibration for every bench capture set; the run trees each carry
    # a byte-identical copy, checked by the GD+CITL pipeline already. Read the
    # GD+CITL master copy.
    path = GDCITL_ROOT / "experiments" / "_rig_calibration" / "calibration.json"
    cal = json.loads(path.read_text())
    if cal.get("mode") != "manual":
        raise ValueError(f"expected manual calibration, got {cal.get('mode')!r}")
    return cal


# --------------------------------------------------------------------------
# Work-list assembly
# --------------------------------------------------------------------------
def build_work() -> list[dict]:
    work: list[dict] = []

    def add(domain, physics, method, target, **kw):
        work.append({"domain": domain, "physics": physics, "method": method,
                     "target": target,
                     "pad_factor": FORWARD_MODELS[physics]["pad_factor"], **kw})

    for physics in ("ideal", "faithful"):
        sim_dirs = sim_comparison_dirs(physics)
        for t, d in sim_dirs.items():
            for m in ("gs", "gs_10000", "gd", "gd_10000",
                      "gs_intensity", "gd_intensity",
                      "transformer_per_target"):
                add("simulation", physics, m, t, recon_npy=sim_recon(d, m))
        for t, p in batched_sim_phase(physics).items():
            add("simulation", physics, "transformer_batched", t, phase_npy=p)

        for t, rec in replay_captures(physics).items():
            for m, cap in rec.items():
                add("bench", physics, m, t, capture=cap, capture_kind="raw")

        for t, rec in replay_converged_captures(physics).items():
            for m, cap in rec.items():
                add("bench_replay", physics, m, t, capture=cap,
                    capture_kind="raw")

        for t, r in sail_runs(physics).items():
            add("bench", physics, "sail", t, capture=r["capture"],
                capture_kind=r["kind"], best_epoch=r["best"])
        for root, name in (("batched_sail", "batched_sail_750"),
                           ("batched_sail_2k", "batched_sail_2000")):
            for t, r in batched_sail_runs(root, physics).items():
                add("bench", physics, name, t, capture=r["capture"],
                    capture_kind=r["kind"], best_epoch=r["best"])

        for init, tag in (("random_init", "random"), ("warm_start", "warm")):
            for alg in ("gd", "gs"):
                for t, r in citl_runs(init, physics, alg).items():
                    add("bench", physics, f"{alg}_citl_{tag}", t,
                        capture=r["capture"], capture_kind=r["kind"],
                        best_iteration=r["best"])

    for t, rec in fno_recons().items():                      # ideal only
        for cond, p in rec.items():
            add("simulation", "ideal", cond, t, recon_npy=p)

    for w in work:
        w["target_image"] = str(target_image(w["target"]) or "")
    return work


def inventory(work: list[dict]) -> None:
    print(f"\n{len(work)} records planned")
    # Provenance guard (2026-08-05): print WHERE the simulation records
    # resolved from, so a stale imported module (or a wrong tree) is visible
    # in the dry run instead of surfacing later as unchanged numbers. After
    # the re-point every entry here must contain '_10k' (plus the FNO run).
    sim_srcs = sorted({Path(w["recon_npy"]).parents[3].name
                       for w in work
                       if w["domain"] == "simulation" and w.get("recon_npy")})
    print("  simulation sources: " + (", ".join(sim_srcs) or "NONE"))
    keys = sorted({(w["domain"], w["physics"], w["method"]) for w in work})
    for k in keys:
        sub = [w for w in work if (w["domain"], w["physics"], w["method"]) == k]
        ok = sum(1 for w in sub
                 if w.get("recon_npy") or w.get("phase_npy") or w.get("capture"))
        miss = sorted(w["target"] for w in sub
                      if not (w.get("recon_npy") or w.get("phase_npy") or w.get("capture")))
        flag = "" if len(sub) == N_TARGETS else f"  <- {len(sub)} targets, expected {N_TARGETS}"
        print(f"  {k[0]:10s} {k[1]:8s} {k[2]:24s} {ok:3d}/{len(sub):3d} inputs{flag}"
              + (f"   MISSING: {', '.join(miss[:6])}{'...' if len(miss) > 6 else ''}"
                 if miss else ""))
    bad = sorted({w["target"] for w in work if not w["target_image"]})
    if bad:
        print(f"  ! no target image for: {', '.join(bad)}")
    fno_faithful = [w for w in work
                    if w["method"].startswith("fno") and w["physics"] == "faithful"]
    assert not fno_faithful, "FNO must be ideal-only by decision"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def build_scorer():
    # evaluate_methods imports raw_camera_processor, which lives in Scripts/ --
    # both directories must be importable, matching the analysis notebooks'
    # Cell 0 convention.
    for sub in ("PyTorchClasses", "Scripts"):
        p = str(REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from evaluate_methods import (compute_metrics, prepare_target_arrays,
                                  _target_eval_for_padding,
                                  reconstruct_from_phase_np)
    try:
        from raw_camera_processor import load_camera_capture_for_citl
    except ImportError:
        from citl_capture import load_camera_capture_for_citl
    import inspect
    if not any(s in inspect.getsource(compute_metrics)
               for s in ("pad_factor ** 2", "pad_factor**2")):
        raise RuntimeError("compute_metrics lacks the resolution-invariant "
                           "support fix; update the repo before scoring.")
    return (compute_metrics, prepare_target_arrays, _target_eval_for_padding,
            reconstruct_from_phase_np, load_camera_capture_for_citl)


def score(work: list[dict], out_path: Path, only: set[str] | None) -> None:
    (compute_metrics, prepare_target, target_for_pad,
     recon_from_phase, load_capture) = build_scorer()
    cal = calibration()
    roi = {k: cal["roi"][k] for k in ("y0", "x0", "h", "w")}
    angle = cal["rotation_deg"]

    from PIL import Image
    records: list[dict] = []
    skipped: list[tuple] = []
    t0 = time.time()
    for w in work:
        prefix = {"simulation": "sim", "bench": "bench",
                  "bench_replay": "replay"}[w["domain"]]
        key = f"{prefix}_{w['method']}"
        if only and key not in only:
            continue
        src = w.get("recon_npy") or w.get("phase_npy") or w.get("capture")
        if not src or not w["target_image"]:
            skipped.append((w["domain"], w["physics"], w["method"],
                            w["target"],
                            "no input file" if not src else "no target image"))
            continue
        pf = w["pad_factor"]
        tgt = prepare_target(image_path=w["target_image"], device="cpu")
        H, W = tgt["H"], tgt["W"]
        target_eval = target_for_pad(tgt["target_eval_np"], pf)

        if w.get("recon_npy"):
            recon = np.load(w["recon_npy"]).astype(np.float32)
        elif w.get("phase_npy"):
            phase = np.load(w["phase_npy"]).astype(np.float32)
            fm = FORWARD_MODELS[w["physics"]]
            recon = recon_from_phase(phase, device="cpu", **fm)
            recon = np.asarray(recon, dtype=np.float32)
        elif w["capture_kind"] == "raw":
            I_cam, *_ = load_capture(
                image_path=str(w["capture"]), roi=roi, out_hw=(H * pf, W * pf),
                device="cpu", angle=angle,
                dc_radius=DC_RADIUS_NATIVE * pf, auto_center=False,
                dc_center=(DC_CENTER_NATIVE[0] * pf, DC_CENTER_NATIVE[1] * pf),
                subtract_min=True)
            recon = I_cam[0].detach().cpu().numpy().astype(np.float32)
        else:   # stored processed capture; resample to the scoring grid if needed
            arr = np.asarray(Image.open(w["capture"]).convert("L"), dtype=np.float32)
            if arr.shape != (H * pf, W * pf):
                import torch
                t = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
                arr = torch.nn.functional.interpolate(
                    t, size=(H * pf, W * pf), mode="bicubic",
                    align_corners=False)[0, 0].numpy().astype(np.float32)
            recon = arr

        m = compute_metrics(recon, target_eval, pf)
        rec = {k: w.get(k) for k in ("domain", "physics", "method", "target",
                                     "pad_factor", "capture_kind",
                                     "best_epoch", "best_iteration")}
        rec = {k: v for k, v in rec.items() if v is not None}
        rec["source_path"] = str(src)
        rec.update(m)
        records.append(rec)
        print(f"  {w['physics']:8s} {w['domain']:10s} {w['method']:24s} "
              f"{w['target']:14s} PSNR={m['psnr']:6.2f} SSIM={m['ssim']:.3f}")

    if skipped:
        print(f"\n** {len(skipped)} planned record(s) were SKIPPED for "
              f"missing inputs -- the scored file is INCOMPLETE: **")
        fam = {}
        for dom, phys, meth, tgt, why in skipped:
            fam.setdefault((dom, phys, meth, why), []).append(tgt)
        for (dom, phys, meth, why), ts in sorted(fam.items()):
            print(f"  {dom:10s} {phys:8s} {meth:24s} {len(ts):2d} target(s), "
                  f"{why}: {', '.join(ts[:5])}{'...' if len(ts) > 5 else ''}")

    timings = collect_timings()
    out = {"meta": {
               "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "gd_iterations": GD_ITER,
               "forward_models": FORWARD_MODELS,
               "dc_center_native": DC_CENTER_NATIVE,
               "dc_radius_native": DC_RADIUS_NATIVE,
               "calibration": {"rotation_deg": angle, "roi": roi},
               "note": ("FNO is ideal-only by decision 2026-08-03. Every "
                        "metric from evaluate_methods.compute_metrics. "
                        "Simulation gs/gd/intensity/transformer_per_target "
                        "are scored from the matched-compute 10k sweep "
                        "(decision 2026-08-05): the same runs whose phases "
                        "the bench replay photographed, one dataset with "
                        "750 read off the sweep."),
           },
           "records": records, "timings": timings}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}  ({len(records)} records, "
          f"{len(timings)} timing rows, {time.time()-t0:.0f}s)")


# --------------------------------------------------------------------------
# Wall-clock collection (best-effort; missing values surface as None and the
# notebook decides what is quotable)
# --------------------------------------------------------------------------
_WALL_RXES = (
    re.compile(r"wall[- ]?clock[:\s]+([\d.]+)\s*(hr|h|hours|min|s|sec)", re.I),
    re.compile(r"[Tt]otal (?:run ?time|time)[:\s]+([\d.]+)\s*(hr|h|hours|min|s|sec)"),
)
_UNIT = {"hr": 3600, "h": 3600, "hours": 3600, "min": 60, "s": 1, "sec": 1}


def _wall_from_log(log: Path) -> float | None:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="replace")
    for rx in _WALL_RXES:
        m = rx.search(txt)
        if m:
            return float(m.group(1)) * _UNIT[m.group(2).lower()]
    return None


def collect_timings() -> list[dict]:
    rows: list[dict] = []
    for physics in ("ideal", "faithful"):
        for t, r in sail_runs(physics).items():
            rows.append({"method": "sail", "physics": physics, "target": t,
                         "seconds": _wall_from_log(r["run_dir"] / "experimental_log.txt")})
        for root, name in (("batched_sail", "batched_sail_750"),
                           ("batched_sail_2k", "batched_sail_2000")):
            base = SA_ROOT / "experiments" / root / physics
            run = _newest(p for p in base.iterdir() if p.is_dir()) if base.exists() else None
            if run is not None:
                rows.append({"method": name, "physics": physics, "target": "ALL_18",
                             "seconds": _wall_from_log(run / "experimental_log.txt")})
        for init, tag in (("random_init", "random"), ("warm_start", "warm")):
            for t, r in citl_runs(init, physics, "gd").items():
                txt = (r["run_dir"] / "experimental_log.txt").read_text(
                    encoding="utf-8", errors="replace")
                m = re.search(r"GD camera-feedback wall-clock:\s*([\d.]+)\s*hr", txt)
                rows.append({"method": f"gd_citl_{tag}", "physics": physics,
                             "target": t,
                             "seconds": float(m.group(1)) * 3600 if m else None})
    return rows


# --------------------------------------------------------------------------
# Scorer verification against the GD+CITL pipeline (notebook section 5)
# --------------------------------------------------------------------------
def score_original_sim_gd() -> dict:
    """Score the PRE-SWEEP comparison's GD recons, verification-only.

    The simulation domain of sail_scored.json reads the 10k sweep
    (decision 2026-08-05), but the GD+CITL paper's scored.json was produced
    from the original simulation_comparison tree. To keep the scorer's
    bit-identity check meaningful, this scores those same original files on
    the fly, through the identical machinery, WITHOUT the results entering
    sail_scored.json. Returns {(physics, target): metrics}.
    """
    (compute_metrics, prepare_target, target_for_pad, *_rest) = build_scorer()
    out: dict = {}
    for physics in ("ideal", "faithful"):
        pf = FORWARD_MODELS[physics]["pad_factor"]
        for t, d in sim_comparison_dirs_original(physics).items():
            p = sim_recon(d, "gd")
            img = target_image(t)
            if p is None or img is None:
                continue
            tgt = prepare_target(image_path=str(img), device="cpu")
            target_eval = target_for_pad(tgt["target_eval_np"], pf)
            recon = np.load(p).astype(np.float32)
            out[(physics, t)] = compute_metrics(recon, target_eval, pf)
    return out


# --------------------------------------------------------------------------
def run(dry_run: bool = False, only=None, out=None) -> list[dict]:
    """Notebook entry point. Returns the work list (dry run) or records.

    only: iterable of '<sim|bench>_<method>' keys to restrict scoring while
    testing, e.g. ("bench_sail", "sim_gd").
    """
    out = Path(out) if out else OUT_DEFAULT
    print(f"results roots:\n  SA     {SA_ROOT}\n  GDCITL {GDCITL_ROOT}\n"
          f"  FNO    {FNO_ROOT}\n  repo   {REPO}\n  out    {out}")
    work = build_work()
    inventory(work)
    if dry_run:
        print("\ndry run - nothing scored")
        return work
    score(work, out, set(only) if only else None)
    return work


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="comma list, e.g. bench_sail,sim_gd")
    a = ap.parse_args()
    run(dry_run=a.dry_run, only=a.only.split(",") if a.only else None, out=a.out)


if __name__ == "__main__":
    main()
