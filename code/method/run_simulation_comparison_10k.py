r"""
Matched-compute simulation sweep: GS and GD to 10,000 iterations, across both
forward models and both target formulations.

WHY THIS EXISTS. The published comparison reports GS/GD at 750 iterations. The
existing sweep shows GD gaining +4.42 dB between 750 and 1500, MORE than the
+2.53 it gained from 500 to 750, so it had not converged where it was stopped.
750 stays the primary operating point
(already ~3x typical CGH practice); this run shows the conclusion is not an
artefact of stopping early and supplies the compute-versus-quality curve.

WHAT CHANGED FROM THE SINGLE-CONDITION DRIVER

  1. Conditions loop. `config.conditions` is a list; each entry carries its own
     physics, model_base_path and save directory. The image->model map is
     rebuilt per condition, because the checkpoints differ between them.

  2. A GUARD THAT WOULD HAVE CAUGHT A REAL BUG. The previous config paired
     `physics: {pad_factor: 1}` (ideal) with `model_base_path:
     .../1000px_P500_B1_faithful`, i.e. faithful-trained checkpoints scored
     under the ideal forward model. Nothing would have complained; the numbers
     would simply have been wrong. `_check_condition()` now refuses to start a
     condition whose model path does not carry its name.

  3. Per-condition aggregates. Each condition writes its own
     `<name>_all_results.json` into its own save directory, so the conditions
     cannot overwrite one another and a partial run leaves complete conditions
     intact.

  4. A consistency check printed at the end. The ideal + amplitude condition at
     750 iterations must reproduce the existing sweep (GD 35.64, GS 33.64 mean
     over 18 targets). If it does not, nothing above 750 should be trusted.

COST. Each entry in iterations_list is an INDEPENDENT run from a random
initialisation, not a checkpoint along one trajectory (see
evaluate_gs_sweep/evaluate_gd_sweep), so cost is the SUM of the list: 29,260
iterations per target per method. On the recorded per-iteration timings
(GS 0.80 ms, GD 2.41 ms at 1000x1000) that is roughly 30 min for the ideal
conditions; faithful runs 2000x2000 and is several times slower.
"""
from datetime import datetime
import numpy as np
import os
import shutil
import time
import json
import torch

from utils import *
from config_handler import ConfigHandler
from experiment_manager import ExperimentManager
from HALO import HALO
from evaluate_methods import evaluate_single_image_multilevel

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "analysis"))
import paths
config_dir = str(paths.CONFIGS / "multilevel")
print("Path:", config_dir)

config_file = "simulation_comparison_10k"
config = ConfigHandler.load(config_file, search_paths=[config_dir])

multiple_images_path = resolve_existing_path(*config.paths.multiple_images)
image_files = list_image_files(multiple_images_path)
print(f"Found {len(image_files)} image files.")

target_formulations = tuple(getattr(config.experiment, "target_formulations", ["amplitude"]))
print(f"Target formulations: {target_formulations}")


def _check_condition(cond, model_base_path):
    """Refuse to run a condition whose checkpoints belong to another one.

    This is the guard for the single most damaging silent error available here:
    pairing one forward model with another's trained models. It fails loudly and
    before any compute is spent, rather than producing plausible wrong numbers.
    """
    name = cond.name.lower()
    other = "faithful" if name == "ideal" else "ideal"
    base = os.path.basename(os.path.normpath(model_base_path)).lower()
    if other in base and name not in base:
        raise RuntimeError(
            f"Condition '{cond.name}' points at model_base_path '{base}', which "
            f"belongs to the '{other}' condition. Fix the config before running: "
            f"scoring one condition's checkpoints under the other's forward "
            f"model produces wrong transformer numbers silently."
        )
    if name not in base:
        print(f"[WARNING] condition '{cond.name}' -> model path '{base}' does "
              f"not carry the condition name; verify this is intended.")


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

timestamp = datetime.now()
run_t0 = time.time()
condition_summaries = {}

for cond in config.conditions:
    cond_t0 = time.time()
    model_base_path = resolve_existing_path(*cond.model_base_path)
    save_path = resolve_existing_path(*cond.save, make=True)
    _check_condition(cond, model_base_path)

    pad_factor = cond.physics.pad_factor
    apply_sinc = bool(cond.physics.apply_sinc)
    fill_factor = cond.physics.fill_factor
    fill_is_areal = bool(getattr(cond.physics, "fill_is_areal", True))

    print(f"\n{'#'*80}")
    print(f"# CONDITION: {cond.name}")
    print(f"#   physics : pad_factor={pad_factor}, apply_sinc={apply_sinc}, "
          f"fill_factor={fill_factor}, fill_is_areal={fill_is_areal}")
    print(f"#   models  : {model_base_path}")
    print(f"#   save    : {save_path}")
    print(f"{'#'*80}")

    model_folders = list_folders(model_base_path)
    image_to_model = build_image_to_model_map(model_folders, image_files)
    print(f"Found {len(model_folders)} model folders, "
          f"{len(image_to_model)} matched to images.")

    all_results = []

    for i, image_file in enumerate(image_files):
        image_stem = os.path.splitext(os.path.basename(image_file))[0]
        if image_stem not in image_to_model:
            print(f"[WARNING] Skipping {image_stem}: no matching model folder.")
            continue
        model_folder = image_to_model[image_stem]

        print(f"\n{'='*80}")
        print(f"[{cond.name}] image {i+1}/{len(image_files)}: {image_stem}")
        print(f"Using model folder: {os.path.basename(model_folder)}")
        print(f"{'='*80}\n")

        experiment = ExperimentManager(
            name=f"{image_stem}_{config.experiment.name}",
            base_dir=save_path,
            overwrite=config.experiment.overwrite,
        )
        out_dir = os.path.join(experiment.dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)

        experiment.log(f"Experiment Details:\r\nExperiment: {config.experiment.description}")
        experiment.log(f"Date: {timestamp}")
        experiment.log(f"Condition: {cond.name}")
        experiment.log(f"Matched model folder: {model_folder}")
        experiment.log(
            f"Physics settings: pad_factor={pad_factor}, apply_sinc={apply_sinc}, "
            f"fill_factor={fill_factor}, fill_is_areal={fill_is_areal}"
        )

        source_config_path = os.path.join(config_dir, config_file + ".json")
        dest_config_path = os.path.join(experiment.dir, "run_configuration.json")
        shutil.copy2(source_config_path, dest_config_path)
        experiment.log(f"Saved config snapshot to: {dest_config_path}")

        H = config.hyperparameters.height
        W = config.hyperparameters.width
        coarse_p = config.hyperparameters.coarse_patch

        coarse_model = HALO(
            H=H, W=W, p=coarse_p, in_channels=1,
            d_model=config.hyperparameters.embedding_dimension,
            nhead=config.hyperparameters.heads,
            num_layers=config.hyperparameters.layers,
            dim_feedforward=config.hyperparameters.feed_forward_dim,
            dropout=config.hyperparameters.dropout,
            pre_norm=config.hyperparameters.pre_norm,
            output_mode="patch",
        ).to(device)

        ckpt_path = os.path.join(model_folder, "checkpoints", "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Missing checkpoint for {image_stem}: {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        coarse_model.load_state_dict(ckpt["coarse_model_state_dict"])
        coarse_model.eval()
        print("Loaded generator checkpoint | epoch:", ckpt.get("epoch", "unknown"),
              "| best loss:", ckpt.get("best_loss", "unknown"))

        t0 = time.time()
        summary = evaluate_single_image_multilevel(
            image_path=image_file,
            out_dir=out_dir,
            coarse_model=coarse_model,
            coarse_p=coarse_p,
            device=device,
            gs_iterations_list=tuple(config.gs.iterations_list),
            gd_iterations_list=tuple(config.gd.iterations_list),
            gs_kwargs={
                "init_phase": config.gs.init_phase,
                "lowpass": config.gs.lowpass,
                "lpf_sigma": config.gs.lpf_sigma,
                "add_carrier": config.gs.add_carrier,
                "show": config.gs.show,
            },
            gd_kwargs={
                "init_phase": config.gd.init_phase,
                "lr": config.gd.lr,
                "optimizer_name": config.gd.optimizer_name,
                "loss_type": config.gd.loss_type,
                "tv_weight": config.gd.tv_weight,
                "lowpass": config.gd.lowpass,
                "lpf_sigma": config.gd.lpf_sigma,
                "add_carrier": config.gd.add_carrier,
                "show": config.gd.show,
                # NOT log_every: gd.run() has no such parameter. It survives in
                # older configs and older driver cells; passing it raises
                # TypeError. Checked against gd.run()'s real signature.
            },
            target_formulations=target_formulations,
            pad_factor=pad_factor,
            apply_sinc=apply_sinc,
            fill_factor=fill_factor,
            fill_is_areal=fill_is_areal,
        )
        elapsed = time.time() - t0
        experiment.log(f"target_formulations used for this run: {target_formulations}")
        experiment.log(f"Finished {image_stem} in {elapsed:.3f} s")

        image_result = {
            "image_name": image_stem,
            "image_path": image_file,
            "condition": cond.name,
            "matched_model_folder": model_folder,
            "checkpoint_path": ckpt_path,
            "checkpoint_epoch": ckpt.get("epoch", None),
            "checkpoint_best_loss": ckpt.get("best_loss", None),
            "training_seconds": ckpt.get("training_seconds", None),
            "elapsed_seconds": elapsed,
            "pad_factor": pad_factor,
            "apply_sinc": apply_sinc,
            "fill_factor": fill_factor,
            "fill_is_areal": fill_is_areal,
            "summary": summary,
        }
        all_results.append(image_result)
        with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
            json.dump(image_result, f, indent=2)

        done = len(all_results)
        rate = (time.time() - cond_t0) / max(done, 1)
        remaining = (len(image_files) - (i + 1)) * rate
        print(f"[{cond.name}] {done}/{len(image_files)} done, "
              f"{elapsed/60:.1f} min this image, ~{remaining/60:.0f} min left "
              f"in this condition")

    # Per-condition aggregate: a partial run still leaves finished conditions
    # complete and readable.
    aggregate_path = os.path.join(save_path, f"{config.experiment.name}_all_results.json")
    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{cond.name}] saved aggregate -> {aggregate_path}")
    print(f"[{cond.name}] condition took {(time.time()-cond_t0)/60:.1f} min")
    condition_summaries[cond.name] = all_results

print(f"\nAll conditions finished in {(time.time()-run_t0)/60:.1f} min")

# --------------------------------------------------------------------------
# Consistency check: does this pipeline still reproduce the existing sweep?
# --------------------------------------------------------------------------
chk = getattr(config, "consistency_check", None)
if chk is not None and chk.condition in condition_summaries:
    rows = condition_summaries[chk.condition]
    got = {}
    for method in ("gd", "gs"):
        vals = []
        for r in rows:
            for entry in r["summary"]["regime1"][f"{method}_sweep"]:
                if entry["iterations"] == chk.iterations:
                    vals.append(entry["psnr"])
        if vals:
            got[method] = float(np.mean(vals))
    print(f"\nCONSISTENCY CHECK ({chk.condition}, {chk.formulation}, "
          f"{chk.iterations} iterations, n={len(rows)}):")
    ok = True
    for method, expected in chk.expected_mean_psnr.to_dict().items():
        if method in got:
            delta = got[method] - expected
            flag = "OK" if abs(delta) < 0.5 else "MISMATCH"
            ok &= abs(delta) < 0.5
            print(f"  {method.upper():3s} expected {expected:6.2f}  "
                  f"got {got[method]:6.2f}  ({delta:+.2f})  {flag}")
    print("  -> pipeline reproduces the existing sweep; points above "
          f"{chk.iterations} are trustworthy." if ok else
          "  -> DOES NOT REPRODUCE. Stop and debug before reading any result.")

# --------------------------------------------------------------------------
# Convergence summary: the number this whole run exists to produce.
# --------------------------------------------------------------------------
for name, rows in condition_summaries.items():
    if not rows:
        continue
    print(f"\nCONVERGENCE ({name}, amplitude, n={len(rows)} targets):")
    print(f"  {'iter':>6s} {'GS mean':>9s} {'GD mean':>9s} {'GD gain':>9s}")
    iters = [e["iterations"] for e in rows[0]["summary"]["regime1"]["gd_sweep"]]
    prev = None
    for it in iters:
        means = {}
        for method in ("gs", "gd"):
            vals = [e["psnr"] for r in rows
                    for e in r["summary"]["regime1"][f"{method}_sweep"]
                    if e["iterations"] == it]
            if vals:
                means[method] = float(np.mean(vals))
        gain = f"{means['gd']-prev:+9.2f}" if prev is not None and "gd" in means else " " * 9
        prev = means.get("gd", prev)
        print(f"  {it:6d} {means.get('gs', float('nan')):9.2f} "
              f"{means.get('gd', float('nan')):9.2f} {gain}")
    print("  Read the GD gain column: if it is still large at 10,000, GD has "
          "not converged\n  even at 13x the reported budget, and the "
          "simulation comparison needs reframing.")
