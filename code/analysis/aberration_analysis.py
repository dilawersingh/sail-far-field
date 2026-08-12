r"""
aberration_analysis.py -- scoring and reporting for the lens-defocus sweep
(bench session 2026-08-05).

Supplies the data behind E9 and behind Fig 5.

THE EXPERIMENT, IN PLAIN TERMS. One target (custom), one forward model
(ideal, matching the submitted Fig 5). SAIL was trained fresh on the aligned
bench in the morning. The Fourier lens was then moved toward the camera in
one direction only, to 0.05, 0.1, 1 and 2 mm, measured with calipers
(nominal values, roughly +/-0.05 mm). At every position three things were
recorded:

  transformer      the simulation-only hologram, which never saw any rig
  sail_saved       the morning's SAIL hologram, adapted to the UNMOVED lens
  sail_retrained   SAIL trained again at that position, which can see the
                   moved lens through the camera

The first two are photographs of a stored hologram and must degrade as the
lens moves, since neither can know the lens moved. The third is the recovery
arm. Moving the lens rather than the camera is equivalent for defocus and
far more repeatable on this rig; the changed SLM-to-lens distance only adds
a quadratic phase factor, invisible in intensity. One direction is
sufficient because intensity blur is symmetric about focus to first order.

TWO DAY-LEVEL CHECKS, both scored here and both required before the curve is
quoted:

  RETURN TO ZERO. At the end of the session the lens was returned to zero and
  the morning's SAIL hologram and the simulation hologram were photographed
  again. Agreement with the morning photographs certifies that nothing else
  in the setup drifted during the day. The bar is the rig's own
  reproducibility (median |delta| 0.14 dB over 306 same-phase pairs, R2.3).

  THE CONVERSE CONTROL (Dilawer's addition). The hologram trained AT 2 mm was
  photographed back at zero. It carries a correction for a defocus that is no
  longer present, so it should be worse at zero than the zero-trained
  hologram is. This is what rules out "the retrained holograms are simply
  better holograms": a hologram that is good at 2 mm and bad at 0 mm can only
  be explained by adaptation to that specific rig state.

ONE PIPELINE. Every capture here is scored with the SAME two functions the
canonical dataset uses: raw_camera_processor.load_camera_capture_for_citl for
the geometry (frozen manual calibration, identical across all five runs, one
alignment all day) and evaluate_methods.compute_metrics for the numbers. The
manual photographs and the training runs' own captures are both raw camera
files and go through an identical path, so they are directly comparable; the
zero-displacement point is photographed twice, once by hand and once by the
training loop, which measures that equivalence rather than assuming it.

CAPTURE INVENTORY (manual photographs, taken with custom's ISO/Tv, camera
untouched all day):
  {condition}/transformer_never_adapted.JPG
  {condition}/sail_stale.JPG           (baseline: sail_baseline_replay.JPG)
  bookend/sail_baseline_from_earlier_run.JPG
  bookend/transformer.JPG
  bookend/sail_trained_at_2mm_shown_at_zero.JPG
and, per condition, the training run's own best-epoch raw frame under
sail_runs/{condition}/ideal/custom_ideal_*/dslr captures/.

Writes analysis/scored_aberration.json. Usage (notebook):
    import aberration_analysis as A
    A.score()      # writes the json
    A.report()     # prints the curve, the day checks and the control
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PHYSICS = "ideal"          # the sweep used the submitted Fig 5's forward model
TARGET = "custom"
PAD_FACTOR = 1             # ideal
DC_CENTER_NATIVE = (495, 510)
DC_RADIUS_NATIVE = 20
NATIVE_HW = (1000, 1000)

# Folder name -> nominal lens displacement toward the camera, mm. Caliper-set,
# roughly +/-0.05 mm; the folder name is the record (no separate log).
CONDITIONS = {
    "baseline": 0.00,
    "lens0p05mm": 0.05,
    "lens0p1mm": 0.10,
    "lens1mm": 1.00,
    "lens2mm": 2.00,
}
ARMS = ("transformer", "sail_saved", "sail_retrained")
ARM_LABEL = {
    "transformer": "Simulation hologram",
    "sail_saved": "SAIL, adapted to aligned bench",
    "sail_retrained": "SAIL, retrained here",
}
LW = 32   # label column width for the printed tables
# Manual-photograph filenames. Baseline names its saved-SAIL photo differently
# because at zero that hologram is the morning's own output rather than a
# stale one; both spellings are accepted so neither has to be renamed.
SAVED_NAMES = ("sail_stale", "sail_baseline_replay")
TRANSFORMER_NAMES = ("transformer_never_adapted", "transformer")
CHECKS = {
    "return_sail_saved": ("bookend", ("sail_baseline_from_earlier_run",
                                      "sail_baseline_replay")),
    "return_transformer": ("bookend", ("transformer",
                                       "transformer_never_adapted")),
    "control_2mm_at_zero": ("bookend", ("sail_trained_at_2mm_shown_at_zero",)),
}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _sa_root() -> Path:
    return Path(os.environ["SAILREV_RESULTS"]) / "Self-Attention" / "multilevel"


def sweep_root() -> Path:
    p = _sa_root() / "experiments" / "aberration_sweep"
    if not p.exists():
        raise FileNotFoundError(f"no aberration_sweep tree at {p}")
    return p


def out_path() -> Path:
    env = os.environ.get("SAILREV_ANALYSIS")
    base = Path(env) if env else (_sa_root() / "analysis")
    base.mkdir(parents=True, exist_ok=True)
    return base / "scored_aberration.json"


def _find(folder: Path, stems) -> Path | None:
    """First existing {stem}.{jpg,JPG,jpeg,png} under folder, in stem order."""
    for stem in stems:
        for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None


_BEST_RX = re.compile(r"[Bb]est\s+epoch[:\s]+(\d+)")


def _retrained_capture(condition: str) -> tuple[Path | None, int | None, Path | None]:
    """(best-epoch raw frame, best epoch, run dir) for one condition's run.

    The training loop keeps exactly first/best/last raw frames, and the run
    summary names the best epoch, so the recovery arm is a raw camera file
    like every other capture here rather than a re-normalised PNG.
    """
    base = sweep_root() / "sail_runs" / condition / PHYSICS
    if not base.exists():
        return None, None, None
    runs = sorted(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith(f"{TARGET}_{PHYSICS}_"))
    if not runs:
        return None, None, None
    run = runs[-1]
    best = None
    for name in ("summary.log", "experimental_log.txt"):
        f = run / name if name != "summary.log" else run.parent / name
        if f.exists():
            hits = _BEST_RX.findall(f.read_text(encoding="utf-8",
                                                errors="replace"))
            if hits:
                best = int(hits[-1])
                break
    raw_dir = run / "dslr captures"
    if best is None and raw_dir.exists():          # first/best/last retained
        eps = sorted(int(m.group(1)) for p in raw_dir.glob("epoch_*.jpg")
                     if (m := re.match(r"epoch_(\d+)\.jpg", p.name)))
        best = eps[1] if len(eps) == 3 else None
    if best is not None:
        p = raw_dir / f"epoch_{best:04d}.jpg"
        if p.exists():
            return p, best, run
    p = run / "outputs" / "best_camera_capture.png"
    return (p if p.exists() else None), best, run


def work_list() -> list[dict]:
    """Every capture to score, with its arm and severity."""
    root = sweep_root()
    work = []
    for cond, mm in sorted(CONDITIONS.items(), key=lambda kv: kv[1]):
        d = root / cond
        if not d.exists():
            print(f"  [skip] no folder for {cond}")
            continue
        p = _find(d, TRANSFORMER_NAMES)
        if p:
            work.append({"arm": "transformer", "condition": cond,
                         "severity_mm": mm, "capture": p, "kind": "manual"})
        p = _find(d, SAVED_NAMES)
        if p:
            work.append({"arm": "sail_saved", "condition": cond,
                         "severity_mm": mm, "capture": p, "kind": "manual"})
        p, best, run = _retrained_capture(cond)
        if p:
            work.append({"arm": "sail_retrained", "condition": cond,
                         "severity_mm": mm, "capture": p,
                         "kind": "training-run", "best_epoch": best,
                         "run_dir": str(run)})
    for name, (folder, stems) in CHECKS.items():
        p = _find(root / folder, stems)
        if p:
            work.append({"arm": name, "condition": folder, "severity_mm": 0.00,
                         "capture": p, "kind": "manual", "check": True})
    return work


# --------------------------------------------------------------------------
# Scoring -- the canonical functions, not reimplementations
# --------------------------------------------------------------------------
def _scorer():
    repo = Path(os.environ.get(
        "SAILREV_REPO",
        Path.home() / "Documents" / "GitHub" / "holography"))
    for sub in ("PythonClasses", "PyTorchClasses", "Scripts"):
        p = str(repo / sub)
        if (repo / sub).exists() and p not in sys.path:
            sys.path.insert(0, p)
    from evaluate_methods import compute_metrics, prepare_target_arrays
    try:
        from raw_camera_processor import load_camera_capture_for_citl
    except ImportError:
        from citl_capture import load_camera_capture_for_citl
    return compute_metrics, prepare_target_arrays, load_camera_capture_for_citl


def _calibration() -> tuple[dict, float]:
    """The frozen manual calibration. Every run of the day wrote its own copy;
    they must all agree, and this asserts it rather than trusting it."""
    cals = sorted(sweep_root().glob("sail_runs/*/_rig_calibration/calibration.json"))
    if not cals:
        raise FileNotFoundError("no calibration.json under the sweep tree")
    texts = {c.read_text() for c in cals}
    if len(texts) != 1:
        raise RuntimeError(f"calibrations differ across the {len(cals)} runs; "
                           "the sweep is not on one alignment")
    cal = json.loads(cals[0].read_text())
    return {k: cal["roi"][k] for k in ("y0", "x0", "h", "w")}, cal["rotation_deg"]


def _target_image() -> Path:
    env = os.environ.get("SAILREV_STOCK")
    stock = (Path(env) if env else
             Path(os.environ["SAILREV_RESULTS"]) / "Stock Images" / "1000x1000")
    for ext in (".png", ".jpg", ".jpeg"):
        p = stock / f"{TARGET}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"target {TARGET} not found under {stock}")


def score(path=None) -> dict:
    compute_metrics, prepare_target, load_capture = _scorer()
    roi, angle = _calibration()
    tgt = prepare_target(image_path=str(_target_image()), device="cpu")
    target_eval = tgt["target_eval_np"]

    work = work_list()
    print(f"aberration | {len(work)} captures, {PHYSICS} model, target "
          f"{TARGET}; calibration frozen and identical across all runs")
    records = []
    for w in work:
        I_cam, *_ = load_capture(
            image_path=str(w["capture"]), roi=roi,
            out_hw=(NATIVE_HW[0] * PAD_FACTOR, NATIVE_HW[1] * PAD_FACTOR),
            device="cpu", angle=angle,
            dc_radius=DC_RADIUS_NATIVE * PAD_FACTOR, auto_center=False,
            dc_center=(DC_CENTER_NATIVE[0] * PAD_FACTOR,
                       DC_CENTER_NATIVE[1] * PAD_FACTOR),
            subtract_min=True)
        recon = I_cam[0].detach().cpu().numpy().astype(np.float32)
        m = compute_metrics(recon, target_eval, PAD_FACTOR)
        rec = {k: w.get(k) for k in ("arm", "condition", "severity_mm", "kind",
                                     "best_epoch", "check")}
        rec = {k: v for k, v in rec.items() if v is not None}
        rec["source_path"] = str(w["capture"])
        rec.update(m)
        records.append(rec)
        print(f"  {w['severity_mm']:5.2f} mm  {w['arm']:22s} "
              f"PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.3f}")

    out = {"meta": {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "physics": PHYSICS, "target": TARGET,
                    "conditions_mm": CONDITIONS,
                    "calibration": {"roi": roi, "rotation_deg": angle},
                    "note": ("Lens moved toward the camera; nominal caliper "
                             "displacements, roughly +/-0.05 mm. Same "
                             "scoring functions as sail_scored.json.")},
           "records": records}
    p = Path(path) if path else out_path()
    p.write_text(json.dumps(out, indent=1))
    print(f"aberration | -> {p}")
    return out


def load(path=None) -> dict:
    p = Path(path) if path else out_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run aberration_analysis.score()")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def curve(d=None) -> dict:
    """{arm: [(mm, psnr, ssim), ...]} sorted by severity, checks excluded."""
    d = d or load()
    out = {a: [] for a in ARMS}
    for r in d["records"]:
        if r.get("check") or r["arm"] not in out:
            continue
        out[r["arm"]].append((r["severity_mm"], r["psnr"], r["ssim"]))
    return {a: sorted(v) for a, v in out.items() if v}


def report(d=None) -> None:
    d = d or load()
    c = curve(d)
    checks = {r["arm"]: r for r in d["records"] if r.get("check")}
    base = {a: v[0] for a, v in c.items() if v and v[0][0] == 0.0}

    print("\nE9 | lens defocus sweep, one target (custom), ideal model, "
          "one alignment\n")
    mms = sorted({m for v in c.values() for m, _, _ in v})
    print(f"  {'arm':{LW}s}" + "".join(f"{m:>9.2f}" for m in mms) + "   mm")
    for a in ARMS:
        if a not in c:
            continue
        by = {m: (p, s) for m, p, s in c[a]}
        print(f"  {ARM_LABEL[a]:{LW}s}" +
              "".join(f"{by[m][0]:9.2f}" if m in by else f"{'--':>9}"
                      for m in mms) + "   PSNR (dB)")
        print(f"  {'':{LW}s}" +
              "".join(f"{by[m][1]:9.3f}" if m in by else f"{'--':>9}"
                      for m in mms) + "   SSIM")

    print("\n  Change from the aligned bench to the largest displacement:")
    for a in ARMS:
        if a in c and len(c[a]) > 1:
            m0, p0, s0 = c[a][0]
            m1, p1, s1 = c[a][-1]
            print(f"    {ARM_LABEL[a]:{LW}s} {p1 - p0:+6.2f} dB, "
                  f"SSIM {s1 - s0:+.3f}  ({m0:.2f} -> {m1:.2f} mm)")

    # The line the figure is really about: what hardware adaptation is WORTH,
    # as a function of how far the hardware has moved from the state it was
    # adapted to. Measured against the arm that never adapted to anything, so
    # it isolates the value of adaptation from the target's own difficulty.
    sim = {m: (p, s) for m, p, s in c.get("transformer", [])}
    print("\n  Value of adaptation (arm minus the simulation hologram):")
    for a in ("sail_saved", "sail_retrained"):
        if a not in c:
            continue
        by = {m: (p, s) for m, p, s in c[a]}
        row_p = "".join(f"{by[m][0] - sim[m][0]:+9.2f}"
                        if m in by and m in sim else f"{'--':>9}" for m in mms)
        row_s = "".join(f"{by[m][1] - sim[m][1]:+9.3f}"
                        if m in by and m in sim else f"{'--':>9}" for m in mms)
        print(f"    {ARM_LABEL[a]:{LW}s}{row_p}   d PSNR (dB)")
        print(f"    {'':{LW}s}{row_s}   d SSIM")

    print("\n  Recovery, retrained minus saved, at each displacement:")
    saved = {m: (p, s) for m, p, s in c.get("sail_saved", [])}
    fresh = {m: (p, s) for m, p, s in c.get("sail_retrained", [])}
    for m in mms:
        if m in saved and m in fresh:
            print(f"    {m:5.2f} mm   {fresh[m][0] - saved[m][0]:+6.2f} dB   "
                  f"SSIM {fresh[m][1] - saved[m][1]:+.3f}")
    if not fresh:
        print("    (no retrained captures found: run score() with the "
              "sail_runs tree present)")

    print("\n  DAY CHECKS")
    for key, arm in (("return_sail_saved", "sail_saved"),
                     ("return_transformer", "transformer")):
        if key in checks and arm in base:
            got = checks[key]
            dp = got["psnr"] - base[arm][1]
            verdict = "OK" if abs(dp) <= 0.30 else "INVESTIGATE"
            print(f"    return to zero, {ARM_LABEL[arm]:{LW}s} "
                  f"{got['psnr']:6.2f} vs {base[arm][1]:6.2f} dB  "
                  f"({dp:+.2f})  {verdict}")
    if "control_2mm_at_zero" in checks and "sail_saved" in base:
        got = checks["control_2mm_at_zero"]
        dp = got["psnr"] - base["sail_saved"][1]
        verdict = "CONTROL HOLDS" if dp < 0 else "CONTROL FAILS"
        print(f"    2 mm-trained hologram shown at zero          "
              f"{got['psnr']:6.2f} vs {base['sail_saved'][1]:6.2f} dB  "
              f"({dp:+.2f})  {verdict}")
        print("      A hologram that is good at 2 mm and worse at zero can "
              "only be explained by\n      adaptation to that rig state, "
              "which is what rules out 'the retrained\n      holograms are "
              "simply better holograms'.")
    print("\n  Reference: the rig reproduces stored holograms to a median "
          "|delta| of 0.14 dB\n  (R2.3, n=306 same-phase pairs), which is the "
          "bar the return-to-zero checks meet.")
    return None
