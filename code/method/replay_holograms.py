r"""
replay_holograms.py

THE capture driver for every experimental replay: display a stored phase
pattern on the SLM, photograph it, move on. No training, no optimisation, no
camera in the loop. Which holograms get captured is decided entirely by the
config, so a new replay is a new config file and never a new script.

    python replay_holograms.py --config simulation_capture           --dry-run
    python replay_holograms.py --config simulation_capture_converged --dry-run

Descended from replay_simulation_holograms.py (sail.ipynb), which produced
replay_simulation_20260724_174615 and the bench columns in the submitted
manuscript. That script hardcoded a four-entry METHOD_SOURCES dict; this one
reads the same information from the config, which is the only structural
change. Everything else it did is kept: the pre-flight that resolves and
shape-checks every phase file before the rig is opened, ExperimentManager for
the run directory and log, the incremental manifest, the ETA line, and the
sync-tolerant staging cleanup.

CONFIGS THAT EXIST

  simulation_capture.json            reproduces the July 2026 run: gs, gd,
                                     transformer_per_target, transformer_batched
                                     at 750 iterations, 2 conditions, 18
                                     targets, 144 captures
  simulation_capture_converged.json  the full re-capture: GS and GD at
                                     both 750 and 10,000 iterations, both
                                     transformers, SAIL, SAIL+, both batched
                                     SAIL budgets and all four GS/GD CITL
                                     arms, 504 captures

HOW A CONFIG DESCRIBES A CAPTURE

Every key in `paths` except capture_root and exposure_settings is a source
tree. Each entry in `run.methods` names an output key, the tree it reads from,
and a glob template in which {condition} and {stem} are substituted:

    {"name": "gd_10000", "base": "sweep_10k",
     "relative": "simulation_comparison_{condition}_10k/{stem}_*/outputs/gd/gd_iter_10000_phase.npy"}

Zero matches is fatal. Several matches is fatal unless the method sets
allow_newest, in which case the most recently modified is used and the choice
is printed in the pre-flight. A method may also carry a "targets" list, which
restricts it to those stems: ablations run on a subset belong in the same
session as everything else without failing the pre-flight elsewhere. The output key is decoupled from the on-disk
folder on purpose: outputs/transformer/ says nothing about how the model was
trained, so it is captured as transformer_per_target.

OUTPUT LAYOUT

    {capture_root}/{run}/{condition}/{stem}/{key}.jpg

which is the shape build_dataset_manifest() in evaluate_methods.py expects.
Point evaluate_experimental_dataset() at {run}/{condition} and it scores these
with no glue code. NOTE: that function has a HARDCODED key list. Any key not in
it is silently skipped.

WHY THE CONVERGED RUN EXISTS

The matched-compute sweep showed GD overtaking the per-target transformer in
ideal simulation at roughly 5,000 iterations, finishing about 4 dB ahead at
10,000, and the manuscript claims compute at matched quality rather than a
superior optimum. What remains to be shown is that the advantage is bought in
simulation and does not survive contact with real optics, because the binding
constraint on hardware is the model-reality gap rather than the strength of the
optimiser. That is only testable if converged GD is photographed on the same
rig, in the same alignment, as everything it is compared against, which is why
the converged config re-captures SAIL and batched SAIL rather than reusing the
July captures.

THREE THINGS THIS ADDS TO THE ORIGINAL

1. SATURATION MEASURED AFTER EVERY CAPTURE. The one addition that affects
   whether the results mean anything. gd_10000 concentrates more energy into
   the signal region than gd_750 and is photographed at an exposure metered for
   gd_750. If it clips, its PSNR is wrong in the direction that flatters the
   argument. Each capture reports the fraction of ROI pixels at or above
   saturation_level; every warning is repeated at the end.

2. RESUME. --resume-dir <run> continues into an existing run directory and
   skips captures already on disk. 324 captures is long enough that a mid-run
   failure should not cost the session.

3. EXPOSURE WRITTEN ONCE PER TARGET. The loop is condition -> target -> method.
   Exposure depends on the target scene and never on which method produced the
   phase, so this records the same thing with 36 exposure writes instead of
   324. The original's CAPTURE_SETTLE_S already made per-capture writes safe;
   this gives the EDSDK fewer chances to return EDS_ERR_DEVICE_BUSY (0x81) over
   a run more than twice as long as the one it was written for.

PRE-FLIGHT, AND WHY IT IS THE IMPORTANT PART

Every phase file is resolved, mmap-loaded and shape-checked, and every target's
exposure is converted to hex, BEFORE the SLM or camera is opened. --dry-run
stops there. Rig time is the scarce resource and discovering a missing
best_phase.npy 200 captures in is the failure worth engineering against. On a
machine where the results tree is cloud-synced with on-demand files the first
dry run is slow, because shape-checking hydrates every placeholder. That is
expected.
"""

from datetime import datetime
import argparse
import glob
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from utils import resolve_existing_path
from config_handler import ConfigHandler
from experiment_manager import ExperimentManager
from hardware import tile_to_slm_centered
from stats_torch import phase_to_uint8
from exposure_lookup import iso_to_hex, tv_to_hex


# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="simulation_capture_converged",
                help="config name in configurations/.../citl (no .json)")
ap.add_argument("--dry-run", action="store_true",
                help="run the pre-flight and stop, touching no hardware")
ap.add_argument("--resume-dir", default=None,
                help="continue into an existing run directory, skipping captures already present")
ap.add_argument("--conditions", nargs="*", default=None)
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--targets", nargs="*", default=None)
args = ap.parse_args()

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "analysis"))
import paths
config_dir = str(paths.CONFIGS / "citl")
config_file = args.config
config = ConfigHandler.load(config_file, search_paths=[config_dir])
source_config_path = os.path.join(config_dir, config_file + ".json")

capture_root = resolve_existing_path(*config.paths.capture_root, make=True)
# Every entry in config.paths other than these two is a SOURCE TREE a method
# can name in its "base" field. Nothing about the trees is hardcoded here, so a
# new capture run is a new config, never a new script.
_NOT_BASES = {"capture_root", "exposure_settings"}
BASES = {k: resolve_existing_path(*v)
         for k, v in config.paths.to_dict().items()
         if k not in _NOT_BASES and not k.startswith("_")}
exposure_settings_path = resolve_existing_path(*config.paths.exposure_settings)

with open(exposure_settings_path) as f:
    exposure = json.load(f)

target_stems = list(exposure["targets"].keys())
subset = args.targets or getattr(config.run, "target_subset", None)
if subset is not None:
    missing = [s for s in subset if s not in target_stems]
    if missing:
        raise ValueError(f"target subset lists unknown targets: {missing}")
    target_stems = list(subset)

METHODS = [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in config.run.methods]
if args.methods:
    METHODS = [m for m in METHODS if m["name"] in args.methods]
CONDITIONS = [c for c in config.run.conditions
              if not args.conditions or c in args.conditions]

SLM_SHAPE = tuple(config.hardware.slm_shape)
SETTLE_TIME_S = config.hardware.settle_time_s
CAPTURE_TIMEOUT_S = config.hardware.capture_timeout_s
# Settle AFTER the capture, before the next exposure write. Without this the
# EDSDK returns EDS_ERR_DEVICE_BUSY (0x81) on the following set_iso/set_tv --
# the same failure that stopped batched SAIL.
CAPTURE_SETTLE_S = getattr(config.hardware, "capture_settle_s", 0.5)
# dll_path is a LIST here, unlike the older configs which hardcoded one user's
# path. The rig machine and the analysis machine are not the same box, and a
# hardcoded EDSDK path fails on whichever one it was not written for.
DLL_PATH = (resolve_existing_path(*config.hardware.dll_path)
            if isinstance(config.hardware.dll_path, list)
            else config.hardware.dll_path)

SAT_LEVEL = getattr(config.hardware, "saturation_level", 250)
SAT_WARN_FRAC = getattr(config.hardware, "saturation_warn_frac", 0.002)

m = config.hardware.manual_alignment
CAMERA_ROI = {"y0": m.y0, "x0": m.x0, "h": m.h, "w": m.w}
ROTATE_DEGREES = m.rotation_deg

H, W = config.run.height, config.run.width

print(f"targets    : {len(target_stems)}")
print(f"methods    : {[x['name'] for x in METHODS]}")
print(f"conditions : {CONDITIONS}")
print(f"total captures: {len(target_stems) * len(METHODS) * len(CONDITIONS)}\n")


# ---------------------------------------------------------------------------
def _resolve(pattern, what, allow_newest=False):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{what}: nothing matches\n    {pattern}")
    if len(hits) > 1:
        if not allow_newest:
            raise RuntimeError(
                f"{what}: {len(hits)} matches, expected 1. Remove stale runs or "
                f"set allow_newest on this method.\n"
                + "\n".join(f"    {h}" for h in hits))
        hits.sort(key=os.path.getmtime)
        return hits[-1], f"newest of {len(hits)}"
    return hits[0], ""


def resolve_phase(method, stem, condition):
    """Locate one saved phase array. Raises rather than returning None: a
    missing phase must stop the pre-flight, not silently drop a capture."""
    base = BASES[method["base"]]
    rel = method["relative"].format(condition=condition, stem=stem)
    return _resolve(os.path.join(base, rel),
                    f"{condition}/{stem}/{method['name']}",
                    allow_newest=bool(method.get("allow_newest")))


def phase_to_slm_frame(phase_np):
    """Identical to the CITL scripts: uint8 quantise -> horizontal flip ->
    centre on the SLM. Any deviation here would make these captures
    incomparable to the CITL ones."""
    phase_8bit = phase_to_uint8(np.asarray(phase_np, dtype=np.float32))
    holo_small = np.fliplr(phase_8bit).copy()
    holo_slm = tile_to_slm_centered(holo_small, slm_shape=SLM_SHAPE)
    return np.ascontiguousarray(holo_slm.astype(np.uint8))


def saturation_fraction(jpg_path):
    """Fraction of ROI pixels at or above SAT_LEVEL, measured on the raw
    capture through the same rotate-then-crop the scorer uses. Deliberately
    not normalised: the question is whether the sensor clipped."""
    gray = np.asarray(Image.open(jpg_path).convert("L"), dtype=np.float32)
    if ROTATE_DEGREES:
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), ROTATE_DEGREES, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    crop = gray[CAMERA_ROI["y0"]:CAMERA_ROI["y0"] + CAMERA_ROI["h"],
                CAMERA_ROI["x0"]:CAMERA_ROI["x0"] + CAMERA_ROI["w"]]
    if crop.size == 0:
        return float("nan")
    return float((crop >= SAT_LEVEL).mean())


# ---------------------------------------------------------------------------
# PRE-FLIGHT: resolve and validate everything before touching hardware.
# ---------------------------------------------------------------------------
print("=" * 74)
print("PRE-FLIGHT -- resolving every phase file before opening the rig")
print("=" * 74)

plan, errors, newest_notes = [], [], []
for condition in CONDITIONS:
    for stem in target_stems:
        for method in METHODS:
            # A method may declare its own target whitelist. Ablations that were
            # only ever run on a couple of targets belong in the same capture
            # session as everything else, but must not fail the pre-flight on
            # the 16 targets they were never run for.
            if method.get("targets") and stem not in method["targets"]:
                continue
            try:
                p, note = resolve_phase(method, stem, condition)
                a = np.load(p, mmap_mode="r")
                if a.shape != (H, W):
                    errors.append(f"{condition}/{stem}/{method['name']}: "
                                  f"shape {a.shape}, expected ({H},{W})")
                    continue
                if note:
                    newest_notes.append(f"{condition}/{stem}/{method['name']}: {note}")
                plan.append({"condition": condition, "stem": stem,
                             "method": method["name"], "phase_path": p})
            except Exception as e:
                errors.append(str(e))

if errors:
    print(f"\n{len(errors)} PROBLEM(S) -- nothing was captured:\n")
    for e in errors:
        print(f"  {e}")
    raise SystemExit(1)

print(f"all {len(plan)} phase files resolved and shape-checked ({H}x{W})")
for condition in CONDITIONS:
    n = sum(1 for r in plan if r["condition"] == condition)
    print(f"  {condition:<10} {n:>4} captures")
for name in [x["name"] for x in METHODS]:
    print(f"    {name:<24} {sum(1 for r in plan if r['method'] == name):>4}")

if newest_notes:
    print(f"\n{len(newest_notes)} source(s) resolved by taking the newest match:")
    for line in newest_notes[:12]:
        print(f"  {line}")
    if len(newest_notes) > 12:
        print(f"  ... and {len(newest_notes) - 12} more")

# Exposure must exist for every target too -- check before the rig is open.
for stem in target_stems:
    e = exposure["targets"][stem]
    iso_to_hex(e["iso"] if e["iso"] is not None else exposure["default_iso"])
    tv_to_hex(e["tv"])
print("exposure settings resolve for every target")

if args.dry_run:
    print("\nDry run: SLM and camera were never opened, nothing captured.")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
if args.resume_dir:
    experiment_dir = args.resume_dir
    if not os.path.isdir(experiment_dir):
        raise SystemExit(f"--resume-dir does not exist: {experiment_dir}")
    log_path = os.path.join(experiment_dir, "experimental_log.txt")

    def experiment_log(line):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(f"\nResuming into {experiment_dir}")
else:
    experiment = ExperimentManager(name="replay_converged", base_dir=capture_root,
                                   overwrite=config.experiment.overwrite)
    experiment_dir = experiment.dir
    experiment_log = experiment.log
    shutil.copy2(source_config_path,
                 os.path.join(experiment_dir, "run_configuration.json"))

experiment_log(f"Date: {datetime.now()}")
experiment_log(f"Targets ({len(target_stems)}): {target_stems}")
experiment_log(f"Methods: {[x['name'] for x in METHODS]} | Conditions: {CONDITIONS}")
experiment_log(f"Alignment: ROI x0={m.x0} y0={m.y0} w={m.w} h={m.h}, rot={m.rotation_deg}")

manifest_path = os.path.join(experiment_dir, "replay_manifest.json")
manifest = {"date": str(datetime.now()), "targets": target_stems,
            "methods": [x["name"] for x in METHODS], "conditions": CONDITIONS,
            "capture_root": capture_root, "captures": []}
if args.resume_dir and os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

slm = camera = None
warnings = []
n_done = n_skip = 0
try:
    from canon_camera import CanonEDSDKCamera
    from holoeye import slmdisplaysdk

    slm = slmdisplaysdk.SLMInstance()
    if not slm.requiresVersion(5):
        raise RuntimeError("Required SLM SDK version not available.")
    err = slm.open()
    if err != slmdisplaysdk.ErrorCode.NoError:
        raise RuntimeError(slm.errorString(err))
    print("SLM opened.")

    camera = CanonEDSDKCamera(
        dll_path=DLL_PATH,
        save_dir=os.path.join(experiment_dir, "_staging"),
        auto_set_save_to_host=True, auto_set_capacity=True, verbose=False)
    camera.initialize()
    camera.open_session()
    camera.assert_manual_mode()
    print("Camera opened, confirmed Manual (M) mode.\n")

    # Alignment is a fixed physical property of the rig; record it alongside
    # the captures so they can be processed later without guessing.
    with open(os.path.join(experiment_dir, "calibration.json"), "w") as f:
        json.dump({"mode": "manual", "roi": CAMERA_ROI,
                   "rotation_deg": ROTATE_DEGREES}, f, indent=2)

    t0 = time.perf_counter()
    i = 0
    # condition -> target -> method, so exposure is written once per target.
    for condition in CONDITIONS:
        for stem in target_stems:
            rows = [r for r in plan
                    if r["condition"] == condition and r["stem"] == stem]
            if not rows:
                continue

            target_dir = os.path.join(experiment_dir, condition, stem)
            os.makedirs(target_dir, exist_ok=True)

            exp_entry = exposure["targets"][stem]
            iso_val = exp_entry["iso"] if exp_entry["iso"] is not None else exposure["default_iso"]
            tv_val = exp_entry["tv"]
            camera.set_iso(iso_to_hex(iso_val))
            camera.set_tv(tv_to_hex(tv_val))
            camera.save_dir = Path(target_dir)
            time.sleep(CAPTURE_SETTLE_S)

            for row in rows:
                i += 1
                method_name = row["method"]
                dest = os.path.join(target_dir, f"{method_name}.jpg")
                if os.path.exists(dest) and not config.experiment.overwrite:
                    n_skip += 1
                    continue

                phase = np.load(row["phase_path"]).astype(np.float32)
                holo = phase_to_slm_frame(phase)

                err = slm.showData(holo)
                if err != slmdisplaysdk.ErrorCode.NoError:
                    raise RuntimeError(
                        f"SLM error on {condition}/{stem}/{method_name}: "
                        f"{slm.errorString(err)}")
                time.sleep(SETTLE_TIME_S)

                result = camera.capture_image(timeout_s=CAPTURE_TIMEOUT_S)
                time.sleep(CAPTURE_SETTLE_S)

                ext = os.path.splitext(result.path)[1] or ".jpg"
                dest = os.path.join(target_dir, f"{method_name}{ext}")
                if os.path.abspath(result.path) != os.path.abspath(dest):
                    if os.path.exists(dest):
                        os.remove(dest)
                    shutil.move(result.path, dest)

                frac = saturation_fraction(dest)
                flag = ""
                if frac == frac and frac > SAT_WARN_FRAC:
                    flag = f"  <-- SATURATION {frac*100:.2f}%"
                    warnings.append((condition, stem, method_name, frac))

                el = time.perf_counter() - t0
                eta = el / max(i - n_skip, 1) * (len(plan) - i)
                print(f"[{i:>3}/{len(plan)}] {condition:<9} {stem:<14} "
                      f"{method_name:<24} ISO {iso_val} Tv {tv_val:<7} "
                      f"sat {frac*100:6.3f}% -> {os.path.basename(dest)}  "
                      f"({el/60:.1f} min, ~{eta/60:.1f} left){flag}")
                experiment_log(
                    f"{condition}/{stem}/{method_name} <- {row['phase_path']} "
                    f"| ISO={iso_val} Tv={tv_val} sat={frac:.6f} -> {dest}")

                manifest["captures"].append({**row, "capture_path": dest,
                                             "iso": iso_val, "tv": tv_val,
                                             "saturation_fraction": frac})
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                n_done += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{n_done} captured, {n_skip} already present, in {elapsed/60:.1f} min")
    experiment_log(f"Complete: {n_done} captured, {n_skip} skipped, "
                   f"{elapsed/60:.1f} min")

finally:
    for obj, name in ((slm, "SLM"), (camera, "Camera")):
        if obj is not None:
            try:
                obj.close()
            except Exception as e:
                print(f"{name} close error (may already be closed):", e)
    print("Closed camera and SLM.")

    if warnings:
        msg = (f"\n{len(warnings)} SATURATION WARNING(S). These captures may be "
               f"clipped and their metrics are not trustworthy:")
        print(msg)
        experiment_log(msg)
        for c, s, meth, fr in warnings:
            line = f"  {c}/{s}/{meth}: {fr*100:.2f}% of ROI at or above {SAT_LEVEL}"
            print(line)
            experiment_log(line)
        tail = ("Re-meter the affected targets before scoring. A clipped "
                "gd_10000 against an unclipped gd_750 is not a comparison.")
        print(tail)
        experiment_log(tail)
    else:
        print("\nNo saturation warnings.")
        experiment_log("No saturation warnings.")

# Cosmetic only. A sync client can hold a handle on a freshly-emptied
# directory on Windows, and a failed tidy-up must never mask a successful run.
staging = os.path.join(experiment_dir, "_staging")
try:
    if os.path.isdir(staging) and not os.listdir(staging):
        os.rmdir(staging)
except OSError as e:
    print(f"(could not remove empty staging dir, harmless: {e})")

print(f"\nartefacts: {experiment_dir}")
print("\nTo score, point evaluate_experimental_dataset() at ONE condition at a time:")
for condition in CONDITIONS:
    print(f"    build_dataset_manifest(target_dir, r\"{os.path.join(experiment_dir, condition)}\")")
print("\nFIRST add these keys to build_dataset_manifest()'s hardcoded key list, "
      "or the captures are silently skipped:")
print("    " + ", ".join(x["name"] for x in METHODS))
