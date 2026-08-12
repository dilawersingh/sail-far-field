"""
evaluate_methods.py - simulation and experimental evaluation harness for GS, GD,
and the self-attention hologram generator (transformer), under the shared
Fraunhofer forward model in physics.py.

This is the single file that produces every quantitative comparison reported in
the manuscript (Fig. 1, Ext. Figs. 1-4, Figs. 2-5, and the experimental
comparisons in Figs. 2-5 / Ext. Fig. 2). All methods are scored identically:
each method's predicted phase (or, for the transformer, predicted field) is
propagated through the same forward model (physics.hologram_intensity_from_phase /
hologram_intensity_from_field), re-normalized identically
(stats_torch.normalize_intensity_sum), and scored with the same metric function
(compute_metrics below) -- so any difference in reported PSNR/SSIM/NMSE reflects a
difference in the predicted phase/field itself, not a difference in how it was
evaluated.

Target formulation:
  Amplitude targets (sqrt(intensity)) for GS/GD are the published formulation;
  the intensity formulation was run as a direct check of whether the choice
  disadvantages those baselines. `evaluate_single_image_multilevel` (the function
  that produces the main GS/GD/transformer comparison) now accepts a
  `target_formulations` argument so it can run GS/GD under either or both
  conditions side by side, without changing what it produces when called the
  original way (see that function's docstring for the exact compatibility
  guarantee). `prepare_target_arrays` and `save_target_artifacts` carry the same
  `target_formulation` parameter through to support this.
"""
import os
import time
import json
import math
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from gs import GerchbergSaxton
from gd import GradientDescentHologram
from physics import hologram_intensity_from_phase, hologram_intensity_from_field
from stats_torch import normalize_intensity_sum, field_to_phase, phase_to_uint8
from patching import patchify, unpatchify
from raw_camera_processor import (
    load_camera_capture_for_citl,
    no_dc_load_camera_capture_for_citl,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)




def save_display_image(arr: np.ndarray, path: str):
    """Min-max normalize to [0,255] for visualization only -- not used for any
    quantitative metric (see compute_metrics, which does its own normalization
    independently for scoring)."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    img = (255.0 * arr).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def _target_eval_for_padding(target_eval_np: np.ndarray, pad_factor: int) -> np.ndarray:
    """
    Return target_eval_np at the resolution compute_metrics needs to match
    recon_np: unchanged if pad_factor==1, bicubic-upsampled to
    (pad_factor*H, pad_factor*W) otherwise. Used identically by
    evaluate_gs_sweep / evaluate_gd_sweep / evaluate_transformer_once so all
    three upsample the SAME way whenever pad_factor>1 -- kept as a single
    helper rather than duplicated inline in three places so they cannot drift.
    """
    if pad_factor <= 1:
        return target_eval_np
    H, W = target_eval_np.shape
    t = torch.from_numpy(target_eval_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t_up = torch.nn.functional.interpolate(
        t, size=(H * pad_factor, W * pad_factor), mode="bicubic", align_corners=False
    )
    return t_up.squeeze(0).squeeze(0).numpy().astype(np.float32)


def compute_metrics(recon: np.ndarray, target: np.ndarray, pad_factor:int=1) -> dict:
    """
    The single scoring function used for every method (GS, GD, transformer)
    in both simulation and experiment. recon/target are independently normalized
    here (both min-max to [0,1] for PSNR/SSIM, both sum-normalized for
    NMSE/diffraction efficiency) -- this function does not assume its inputs
    arrive pre-normalized in any particular convention, so it produces consistent
    metrics regardless of which upstream normalization a given caller used before
    this point.
    """
    recon = np.asarray(recon, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    recon_01 = recon - recon.min()
    recon_01 = recon_01 / (recon_01.max() + 1e-12)

    target_01 = target - target.min()
    target_01 = target_01 / (target_01.max() + 1e-12)

    recon_sum = np.clip(recon, 0.0, None)
    recon_sum = recon_sum / (recon_sum.sum() + 1e-12)

    target_sum = np.clip(target, 0.0, None)
    target_sum = target_sum / (target_sum.sum() + 1e-12)

    psnr = peak_signal_noise_ratio(target_01, recon_01, data_range=1.0)
    ssim = structural_similarity(target_01, recon_01, data_range=1.0)
    mse = np.mean((recon_01 - target_01) ** 2)
    nmse = np.sum((recon_01 - target_01) ** 2) / (np.sum(target_01 ** 2) + 1e-12)

    support = target_sum > (1e-6 / (pad_factor ** 2))
    eff = recon_sum[support].sum() / (recon_sum.sum() + 1e-12)

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "mse": float(mse),
        "nmse": float(nmse),
        "diffraction_efficiency": float(eff),
    }


def reconstruct_from_phase_np(phase_np: np.ndarray, device: str,
                              pad_factor: int = 1, apply_sinc: bool = False,
                              fill_factor: float = 1.0, fill_is_areal: bool = True) -> np.ndarray:
    """
    Re-propagate a phase array through the shared forward model and re-normalize
    independently of however the optimizer that produced it normalized things
    internally. This is what makes GS, GD, and the transformer comparable: all
    three end up here before compute_metrics is called on their output, so the
    metrics reflect the predicted phase itself, not each method's internal
    training/optimization bookkeeping.

    pad_factor/apply_sinc/fill_factor/fill_is_areal are forwarded, UNCHANGED IN
    MEANING, to physics.hologram_intensity_from_phase. Defaults (1, False, 1.0,
    True) reproduce the original byte-identical ideal-FFT scoring. IMPORTANT:
    callers (evaluate_gs_sweep/evaluate_gd_sweep/evaluate_transformer_once) pass
    the SAME settings used to GENERATE the phase, so generation and scoring are
    always on the same forward model -- never call this with different settings
    than whatever produced phase_np, or the reported metric will not reflect
    what was actually optimized/simulated.
    """
    phase_t = torch.from_numpy(phase_np.astype(np.float32)).unsqueeze(0).to(device)
    recon_t = hologram_intensity_from_phase(
        phase_t, pad_factor=pad_factor, apply_sinc=apply_sinc,
        fill_factor=fill_factor, fill_is_areal=fill_is_areal,
    )
    recon_t = normalize_intensity_sum(recon_t)
    return recon_t[0].detach().cpu().numpy().astype(np.float32)


def run_transformer_method(
    xb: torch.Tensor,
    coarse_model,
    coarse_p: int,
    eps_field: float = 1e-6,
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    Single forward pass of the self-attention hologram generator. Unlike GS/GD,
    there is no target_formulation concept here: the transformer consumes the raw
    target intensity image directly (it predicts a phase/field, it does not
    iteratively enforce a magnitude constraint the way GS does, nor train against
    an explicit target the way GD does per-call) -- so this function, and its
    caller evaluate_transformer_once, are unaffected by the GS/GD target
    formulation experiment and do not need an equivalent parameter.

    pad_factor/apply_sinc/fill_factor/fill_is_areal are forwarded, UNCHANGED IN
    MEANING, to physics.hologram_intensity_from_field. Defaults (1, False, 1.0,
    True) reproduce the original byte-identical ideal-FFT reconstruction. Note:
    this only changes how the transformer's OWN predicted field is propagated
    for scoring -- the transformer's field prediction itself is unaffected
    (it was trained/predicted independently of this choice).
    """
    coarse_model.eval()

    with torch.no_grad():
        B, H, W = xb.shape

        X_coarse = patchify(xb, coarse_p)
        y_coarse = coarse_model(X_coarse)
        Y = unpatchify(y_coarse, H, W, coarse_p, C=2)

        I_pred = hologram_intensity_from_field(
            Y, eps=eps_field, pad_factor=pad_factor, apply_sinc=apply_sinc,
            fill_factor=fill_factor, fill_is_areal=fill_is_areal,
        )
        phase = field_to_phase(Y)[0]

        return {
            "field": Y,
            "phase_radians": phase.detach().cpu().numpy(),
            "phase_uint8": phase_to_uint8(phase),
            "recon_intensity": I_pred[0].detach().cpu().numpy().astype(np.float32),
        }


def prepare_target_arrays(image_path: str, device: str, target_formulation: str = "amplitude"):
    """
    Build the optimization target and the (independent) evaluation target for a
    single image.

    target_formulation controls ONLY what is returned as "target_optim_np" below
    -- the array that GS/GD will use to drive their optimization:
        - "amplitude" (default, original/published behaviour):
              target_optim_np = sqrt(clip(X, 0))
              This is the exact formulation used for all results reported in the
              submitted manuscript (Fig. 1, Ext. Figs. 1-4, Figs. 2-5).
        - "intensity" (the target-formulation check of whether amplitude-
          vs. intensity-domain optimization targets disadvantage the GS/GD
          baselines):
              target_optim_np = clip(X, 0)
              GS's replay-plane magnitude constraint and GD's intensity training
              target (via target_formulation plumbed through to gd.run(), see
              gd.py) are then driven directly by intensity rather than amplitude.

    target_eval_np (used for ALL scoring/metrics, regardless of formulation) is
    deliberately computed from the raw image X directly and never from
    target_optim_np, so switching target_formulation cannot change what any
    method is scored against -- only what GS/GD optimize toward. This
    independence was verified by trace and by a standalone numerical check before
    this parameter was added.

    Returns a dict containing "target_optim_np" (formulation-aware) and, for
    backward compatibility with any existing external scripts/notebooks that read
    "target_amplitude_np" directly, an alias that is only present when
    target_formulation == "amplitude". New code should read "target_optim_np".
    """
    if target_formulation not in ("amplitude", "intensity"):
        raise ValueError(
            f"Unknown target_formulation: {target_formulation!r}. "
            f"Expected 'amplitude' (published/default) or 'intensity'."
        )

    img = Image.open(image_path).convert("L")
    X = np.asarray(img, dtype=np.float32)

    H, W = X.shape
    X_batched = X.reshape(1, H, W)

    X_clipped = np.clip(X, 0.0, None)
    if target_formulation == "amplitude":
        target_optim_np = np.sqrt(X_clipped).astype(np.float32)
    else:  # "intensity"
        target_optim_np = X_clipped.astype(np.float32)

    # Evaluation target: independent of target_formulation by construction (see
    # docstring above) -- computed from X directly, never from target_optim_np.
    target_t = torch.from_numpy(X_batched).to(device=device, dtype=torch.float32)
    target_t = normalize_intensity_sum(target_t)
    target_eval_np = target_t[0].detach().cpu().numpy().astype(np.float32)

    out = {
        "raw_intensity_batched": X_batched,
        "target_optim_np": target_optim_np,
        "target_formulation": target_formulation,
        "target_eval_np": target_eval_np,
        "H": H,
        "W": W,
    }
    if target_formulation == "amplitude":
        # Back-compat alias only -- intentionally absent for "intensity" so that
        # any old code path assuming "target_amplitude_np" is amplitude-shaped
        # fails loudly (KeyError) rather than silently treating an intensity
        # array as if it were amplitude.
        out["target_amplitude_np"] = target_optim_np

    return out


def save_target_artifacts(out_dir: str, target_eval_np: np.ndarray, target_optim_np: np.ndarray,
                           target_formulation: str = "amplitude"):
    """
    target_formulation controls the saved filename for the optimization target
    array:
        - "amplitude": writes "target_amplitude.npy" -- UNCHANGED filename,
          preserves exact on-disk naming from the published pipeline.
        - "intensity": writes "target_intensity_optim.npy" -- a new, distinct
          name (deliberately NOT "target_intensity.npy", which is already used
          below for the evaluation target and must never be overwritten by this
          function regardless of target_formulation).
    """
    np.save(os.path.join(out_dir, "target_intensity.npy"), target_eval_np)
    optim_filename = "target_amplitude.npy" if target_formulation == "amplitude" else "target_intensity_optim.npy"
    np.save(os.path.join(out_dir, optim_filename), target_optim_np)
    save_display_image(target_eval_np, os.path.join(out_dir, "target.png"))


def evaluate_gs_sweep(
    target_amplitude_np,
    target_eval_np,
    out_dir,
    device,
    iterations_list,
    binary_phase: bool = False,
    gs_kwargs=None,
    target_formulation: str = "amplitude",
):
    """
    GS does not assume a domain for its target (see gs.py module docstring) --
    target_formulation here is used ONLY to tag output filenames distinctly when
    both formulations are swept into related directories (see
    evaluate_single_image_multilevel), so that an "amplitude" sweep and an
    "intensity" sweep writing into sibling directories cannot collide even if a
    caller passes the same out_dir by mistake. It is NOT passed to gs.run(),
    since gs.run() needs no such parameter -- it treats target_amplitude_np as a
    raw magnitude constraint regardless of what it represents.
    """
    ensure_dir(out_dir)
    gs_kwargs = gs_kwargs or {}
    # Extract the faithful-model settings from gs_kwargs (already splatted into
    # gs.run() below for GENERATION) so the EXACT SAME settings are reused for
    # SCORING (reconstruct_from_phase_np). This guarantees generation and
    # scoring can never silently mismatch. Defaults (1, False, 1.0, True)
    # reproduce the original ideal-FFT behaviour when absent from gs_kwargs.
    pad_factor = gs_kwargs.get("pad_factor", 1)
    apply_sinc = gs_kwargs.get("apply_sinc", False)
    fill_factor = gs_kwargs.get("fill_factor", 1.0)
    fill_is_areal = gs_kwargs.get("fill_is_areal", True)
    target_eval_np_scaled = _target_eval_for_padding(target_eval_np, pad_factor)

    gs = GerchbergSaxton()
    results = []
    tag_suffix = "" if target_formulation == "amplitude" else f"_{target_formulation}"

    for n_iter in iterations_list:
        t0 = time.time()
        out = gs.run(
            target_amplitude_np,
            binary_phase=binary_phase,
            iterations=n_iter,
            device=device,
            **gs_kwargs,
        )
        elapsed = time.time() - t0

        recon_np = reconstruct_from_phase_np(
            out["phase_radians"], device=device,
            pad_factor=pad_factor, apply_sinc=apply_sinc,
            fill_factor=fill_factor, fill_is_areal=fill_is_areal,
        )
        metrics = compute_metrics(recon_np, target_eval_np_scaled, pad_factor)

        item = {
            "iterations": int(n_iter),
            "wall_clock_seconds": float(elapsed),
            "target_formulation": target_formulation,
            **metrics,
        }
        results.append(item)

        tag = f"gs_iter_{n_iter}{tag_suffix}"
        np.save(os.path.join(out_dir, f"{tag}_phase.npy"), out["phase_radians"])
        Image.fromarray(out["phase_uint8"]).save(os.path.join(out_dir, f"{tag}_phase_8bit.png"))
        np.save(os.path.join(out_dir, f"{tag}_recon.npy"), recon_np)
        save_display_image(recon_np, os.path.join(out_dir, f"{tag}_recon.png"))

    return results


def evaluate_gd_sweep(
    target_amplitude_np,
    target_eval_np,
    out_dir,
    device,
    binary_phase,
    iterations_list,
    gd_kwargs=None,
    target_formulation: str = "amplitude",
):
    """
    Unlike GS, GD's run() DOES need to know target_formulation explicitly (see
    gd.py module docstring: GD historically always squared its target array,
    assuming amplitude; this is only correct when target_formulation=="amplitude").
    target_formulation is therefore both (a) forwarded into gd.run() so the
    correct intensity target is used internally, and (b) used to tag output
    filenames, same as evaluate_gs_sweep, so sibling-directory sweeps under
    different formulations cannot collide.

    NOTE: if gd_kwargs already contains a "target_formulation" key, that value
    takes precedence (gd_kwargs is expanded after the explicit kwarg below would
    be -- see call below); this function's target_formulation parameter exists
    so callers that don't already build a full gd_kwargs dict have a simple way
    to set it.
    """
    ensure_dir(out_dir)
    gd_kwargs = dict(gd_kwargs or {})
    gd_kwargs.setdefault("target_formulation", target_formulation)
    # Extract the faithful-model settings from gd_kwargs (already splatted into
    # gd.run() below for GENERATION) so the EXACT SAME settings are reused for
    # SCORING (reconstruct_from_phase_np) -- mirrors evaluate_gs_sweep exactly.
    pad_factor = gd_kwargs.get("pad_factor", 1)
    apply_sinc = gd_kwargs.get("apply_sinc", False)
    fill_factor = gd_kwargs.get("fill_factor", 1.0)
    fill_is_areal = gd_kwargs.get("fill_is_areal", True)
    target_eval_np_scaled = _target_eval_for_padding(target_eval_np, pad_factor)

    gd = GradientDescentHologram()
    results = []
    tag_suffix = "" if target_formulation == "amplitude" else f"_{target_formulation}"

    for n_iter in iterations_list:
        t0 = time.time()
        out = gd.run(
            target_amplitude_np,
            iterations=n_iter,
            binary_phase=binary_phase,
            device=device,
            **gd_kwargs,
        )
        elapsed = time.time() - t0

        recon_np = reconstruct_from_phase_np(
            out["phase_radians"], device=device,
            pad_factor=pad_factor, apply_sinc=apply_sinc,
            fill_factor=fill_factor, fill_is_areal=fill_is_areal,
        )
        metrics = compute_metrics(recon_np, target_eval_np_scaled, pad_factor)

        item = {
            "iterations": int(n_iter),
            "wall_clock_seconds": float(elapsed),
            "best_loss": float(out.get("best_loss", math.nan)),
            "target_formulation": target_formulation,
            **metrics,
        }
        results.append(item)

        tag = f"gd_iter_{n_iter}{tag_suffix}"
        np.save(os.path.join(out_dir, f"{tag}_phase.npy"), out["phase_radians"])
        Image.fromarray(out["phase_uint8"]).save(os.path.join(out_dir, f"{tag}_phase_8bit.png"))
        np.save(os.path.join(out_dir, f"{tag}_recon.npy"), recon_np)
        save_display_image(recon_np, os.path.join(out_dir, f"{tag}_recon.png"))

    return results




def evaluate_transformer_once(
    target_intensity_np,
    target_eval_np,
    out_dir,
    coarse_model,
    coarse_p,
    device,
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    pad_factor/apply_sinc/fill_factor/fill_is_areal: forwarded to
    run_transformer_method (generation AND scoring share these, since the
    transformer's forward pass and its scoring reconstruction are the same
    call -- see run_transformer_method). Defaults (1, False, 1.0, True)
    reproduce the original byte-identical behaviour.
    """
    ensure_dir(out_dir)

    xb = torch.from_numpy(target_intensity_np).to(device=device, dtype=torch.float32)
    target_eval_np_scaled = _target_eval_for_padding(target_eval_np, pad_factor)

    t0 = time.time()
    out = run_transformer_method(
        xb=xb,
        coarse_model=coarse_model,
        coarse_p=coarse_p,
        pad_factor=pad_factor,
        apply_sinc=apply_sinc,
        fill_factor=fill_factor,
        fill_is_areal=fill_is_areal,
    )
    elapsed = time.time() - t0

    recon_np = np.asarray(out["recon_intensity"], dtype=np.float32)
    recon_np = recon_np / (recon_np.sum() + 1e-12)

    metrics = compute_metrics(recon_np, target_eval_np_scaled, pad_factor)

    result = {
        "wall_clock_seconds": float(elapsed),
        **metrics,
    }

    np.save(os.path.join(out_dir, "transformer_phase.npy"), out["phase_radians"])
    Image.fromarray(out["phase_uint8"]).save(os.path.join(out_dir, "transformer_phase_8bit.png"))
    np.save(os.path.join(out_dir, "transformer_recon.npy"), recon_np)
    save_display_image(recon_np, os.path.join(out_dir, "transformer_recon.png"))

    return result






def evaluate_single_image_multilevel(
    image_path: str,
    out_dir: str,
    coarse_model,
    coarse_p: int,
    device="cuda",
    gs_iterations_list=(10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000),
    gd_iterations_list=(100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000),
    gs_kwargs=None,
    gd_kwargs=None,
    target_formulations=("amplitude",),
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    Run the full GS / GD / transformer simulation comparison for one image.

    target_formulations: tuple of one or more of "amplitude" | "intensity".

    pad_factor / apply_sinc / fill_factor / fill_is_areal: single top-level
    knob for the physically-faithful forward model (see physics.py), applied
    consistently across GS, GD, AND the transformer. Defaults (1, False, 1.0,
    True) reproduce the exact ideal-FFT behaviour described in the
    COMPATIBILITY GUARANTEE below. These four are injected into gs_kwargs and
    gd_kwargs via .setdefault() (so an explicit pad_factor/etc. already present
    in a caller-supplied gs_kwargs/gd_kwargs dict takes precedence -- same
    override convention already used for target_formulation in gd_kwargs) and
    passed directly to evaluate_transformer_once (which has no kwargs dict of
    its own). This is the ONE place a caller needs to set these for all three
    methods to use the SAME forward model -- do not set pad_factor separately
    inside a custom gs_kwargs/gd_kwargs unless you specifically want GS/GD and
    the transformer to use DIFFERENT forward models (unusual; not needed for
    the standard faithful-vs-ideal 2x2 ablation).

    COMPATIBILITY GUARANTEE: called with the default target_formulations=("amplitude",)
    (i.e. exactly as every existing call site in this codebase calls it today, since
    none of them currently pass this argument), this function is byte-identical in
    behaviour to the version that produced every published GS/GD/transformer
    simulation result (Fig. 1, Ext. Figs. 1-4): same output paths
    (out_dir/gs/gs_iter_{n}_*, out_dir/gd/gd_iter_{n}_*, out_dir/transformer/*),
    same summary dict shape and keys, same target_amplitude.npy filename. This was
    verified directly (numerical diff against the unmodified function on a
    synthetic target, not just inspected) before this parameter was added -- see
    dev-notes.

    Passing additional formulations, e.g. target_formulations=("amplitude",
    "intensity"), runs GS/GD again under each additional formulation and writes
    those results to a namespaced sibling subdirectory
    (out_dir/gs_intensity_target/, out_dir/gd_intensity_target/) -- never into the
    bare out_dir/gs/ or out_dir/gd/ used by the first (always default-formulation)
    sweep, so the original sweep's files cannot be touched regardless of how many
    additional formulations are requested. The transformer is run exactly once
    per call regardless of len(target_formulations), since it does not depend on
    target_formulation (see run_transformer_method docstring).

    summary["regime1"]["gs_sweep"] / ["gd_sweep"] always refer to the FIRST
    formulation in target_formulations (== "amplitude" under the default, hence
    identical to the pre-existing meaning of these keys). Results for any
    additional formulations are reported separately under
    summary["regime1"]["additional_target_formulations"][<formulation>], so that
    nothing reads as ambiguous about which result corresponds to the published
    condition.
    """
    ensure_dir(out_dir)

    if not target_formulations:
        raise ValueError("target_formulations must contain at least one formulation.")
    for f in target_formulations:
        if f not in ("amplitude", "intensity"):
            raise ValueError(f"Unknown target_formulation: {f!r} in target_formulations={target_formulations!r}")
    if len(set(target_formulations)) != len(target_formulations):
        raise ValueError(f"target_formulations contains duplicates: {target_formulations!r}")

    primary_formulation = target_formulations[0]
    additional_formulations = target_formulations[1:]

    # ---- Resolve gs_kwargs/gd_kwargs ONCE, injecting the top-level
    # pad_factor/apply_sinc/fill_factor/fill_is_areal via .setdefault() (an
    # explicit value already present in a caller-supplied dict wins -- same
    # precedent as target_formulation's existing .setdefault() in gd_kwargs).
    # Built once and reused for BOTH the primary formulation and every entry
    # in additional_formulations, so all sweeps in this call share the SAME
    # forward-model settings unless a caller went out of their way to build
    # per-call dicts with conflicting values.
    gs_kwargs_resolved = dict(gs_kwargs) if gs_kwargs else {
        "init_phase": "random",
        "lowpass": False,
        "lpf_sigma": 0.08,
        "add_carrier": False,
        "show": False,
    }
    gs_kwargs_resolved.setdefault("pad_factor", pad_factor)
    gs_kwargs_resolved.setdefault("apply_sinc", apply_sinc)
    gs_kwargs_resolved.setdefault("fill_factor", fill_factor)
    gs_kwargs_resolved.setdefault("fill_is_areal", fill_is_areal)

    gd_kwargs_resolved = dict(gd_kwargs) if gd_kwargs else {
        "init_phase": "random",
        "lr": 0.03,
        "optimizer_name": "adam",
        "loss_type": "mse_sum",  # see gd.py: this is what the code has always actually done
        "tv_weight": 0.0,
        "lowpass": False,
        "lpf_sigma": 0.08,
        "add_carrier": False,
        "show": False,
    }
    gd_kwargs_resolved.setdefault("pad_factor", pad_factor)
    gd_kwargs_resolved.setdefault("apply_sinc", apply_sinc)
    gd_kwargs_resolved.setdefault("fill_factor", fill_factor)
    gd_kwargs_resolved.setdefault("fill_is_areal", fill_is_areal)

    # ---- Primary (default-path) formulation: writes to the ORIGINAL, unchanged
    # paths. With target_formulations=("amplitude",) (the default), everything
    # below this point through the transformer call is identical to the
    # pre-existing function body.
    target = prepare_target_arrays(image_path=image_path, device=device, target_formulation=primary_formulation)
    X = target["raw_intensity_batched"]
    target_optim_np = target["target_optim_np"]
    target_eval_np = target["target_eval_np"]
    H, W = target["H"], target["W"]

    save_target_artifacts(out_dir, target_eval_np, target_optim_np, target_formulation=primary_formulation)

    summary = {
        "image_path": image_path,
        "slm_shape": [H, W],
        "comparison_mode": "multilevel",
        "target_formulation": primary_formulation,
        "regime1": {},
    }

    gs_dir = os.path.join(out_dir, "gs")
    gs_results = evaluate_gs_sweep(
        target_amplitude_np=target_optim_np,
        target_eval_np=target_eval_np,
        out_dir=gs_dir,
        device=device,
        binary_phase=False,
        iterations_list=gs_iterations_list,
        gs_kwargs=gs_kwargs_resolved,
        target_formulation=primary_formulation,
    )
    summary["regime1"]["gs_sweep"] = gs_results

    gd_dir = os.path.join(out_dir, "gd")
    gd_results = evaluate_gd_sweep(
        target_amplitude_np=target_optim_np,
        target_eval_np=target_eval_np,
        out_dir=gd_dir,
        device=device,
        binary_phase=False,
        iterations_list=gd_iterations_list,
        gd_kwargs=gd_kwargs_resolved,
        target_formulation=primary_formulation,
    )
    summary["regime1"]["gd_sweep"] = gd_results

    # Transformer: run exactly once, regardless of how many target_formulations
    # were requested -- see docstring. Uses the raw intensity image directly, not
    # target_optim_np, so it is unaffected by target_formulation entirely.
    tf_dir = os.path.join(out_dir, "transformer")
    transformer_result = evaluate_transformer_once(
        target_intensity_np=X,
        target_eval_np=target_eval_np,
        out_dir=tf_dir,
        coarse_model=coarse_model,
        coarse_p=coarse_p,
        device=device,
        pad_factor=pad_factor,
        apply_sinc=apply_sinc,
        fill_factor=fill_factor,
        fill_is_areal=fill_is_areal,
    )
    summary["regime1"]["transformer"] = transformer_result

    # ---- Additional formulations (only entered if target_formulations has more
    # than one entry -- never executes under the default call). Each additional
    # formulation writes to namespaced sibling directories that cannot collide
    # with the primary formulation's gs/, gd/, or transformer/ directories above,
    # or with each other.
    if additional_formulations:
        summary["regime1"]["additional_target_formulations"] = {}

    for formulation in additional_formulations:
        formulation_target = prepare_target_arrays(
            image_path=image_path, device=device, target_formulation=formulation
        )
        formulation_optim_np = formulation_target["target_optim_np"]
        # target_eval_np is identical regardless of formulation (see
        # prepare_target_arrays docstring) -- re-using the primary one here is
        # equivalent to recomputing it, included for clarity/robustness rather
        # than relying on that invariant silently.
        formulation_eval_np = formulation_target["target_eval_np"]

        formulation_subdir = os.path.join(out_dir, f"_{formulation}_target")
        ensure_dir(formulation_subdir)
        save_target_artifacts(
            formulation_subdir, formulation_eval_np, formulation_optim_np, target_formulation=formulation
        )

        gs_results_f = evaluate_gs_sweep(
            target_amplitude_np=formulation_optim_np,
            target_eval_np=formulation_eval_np,
            out_dir=os.path.join(out_dir, f"gs_{formulation}_target"),
            device=device,
            binary_phase=False,
            iterations_list=gs_iterations_list,
            gs_kwargs=gs_kwargs_resolved,
            target_formulation=formulation,
        )

        gd_results_f = evaluate_gd_sweep(
            target_amplitude_np=formulation_optim_np,
            target_eval_np=formulation_eval_np,
            out_dir=os.path.join(out_dir, f"gd_{formulation}_target"),
            device=device,
            binary_phase=False,
            iterations_list=gd_iterations_list,
            gd_kwargs=gd_kwargs_resolved,
            target_formulation=formulation,
        )

        summary["regime1"]["additional_target_formulations"][formulation] = {
            "gs_sweep": gs_results_f,
            "gd_sweep": gd_results_f,
        }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary



#---------------------------------------------------
# EXPERIMENTAL COMPARISON METHODS
#---------------------------------------------------
# Kept only so a caller can reproduce the pre-2026-08 behaviour exactly by
# passing keys=LEGACY_CAPTURE_KEYS. Nothing should add to this list.
LEGACY_CAPTURE_KEYS = [
    "gs",
    "gd",
    "gd_citl",
    "transformer_per_target",     # was "transformer"
    "transformer_batched",        # was "shared_model"
    "citl",
    "citl_corrector",
    "transformer_corrector",
]


def build_dataset_manifest(target_dir, capture_root, keys=None, verbose=True):
    """Pair each target image with the captures sitting under capture_root.

    keys=None (the default) DISCOVERS the method keys from disk: every
    {capture_root}/{image_name}/{key}.jpg becomes a method named {key}.

    This used to be a hardcoded list, which meant every new capture run
    silently dropped any method not already in it. That is the wrong failure
    mode: a missing column looks exactly like a method that scored nothing, and
    nothing in the pipeline complains. The converged replay adds fourteen keys
    at once, and the next run will add more, so discovery replaces the list.

    Pass keys=[...] to pin an explicit set, or keys=LEGACY_CAPTURE_KEYS to
    reproduce the old behaviour exactly. Names beginning with "_" are ignored
    so staging and scratch files cannot become methods.

    evaluate_single_target_experimental() already routes any key outside its
    known_methods set through the generic loop, which does identical work, so
    discovered keys need no further registration.
    """
    manifest = []
    discovered = set()

    for fname in sorted(os.listdir(target_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_name = os.path.splitext(fname)[0]

        target_path = os.path.join(target_dir, fname)
        capture_dir = os.path.join(capture_root, image_name)

        capture_paths = {}

        if keys is None:
            if os.path.isdir(capture_dir):
                for cap in sorted(os.listdir(capture_dir)):
                    stem, ext = os.path.splitext(cap)
                    if ext.lower() not in (".jpg", ".jpeg"):
                        continue
                    if stem.startswith("_"):
                        continue
                    capture_paths[stem] = os.path.join(capture_dir, cap)
                    discovered.add(stem)
        else:
            for key in keys:
                path = os.path.join(capture_dir, f"{key}.jpg")
                if os.path.exists(path):
                    capture_paths[key] = path
                    discovered.add(key)

        manifest.append({
            "image_name": image_name,
            "image_path": target_path,
            "capture_paths": capture_paths
        })

    if verbose:
        counts = {k: sum(1 for m in manifest if k in m["capture_paths"])
                  for k in sorted(discovered)}
        mode = "discovered" if keys is None else "requested"
        print(f"build_dataset_manifest: {len(manifest)} targets, "
              f"{len(counts)} methods {mode} in {capture_root}")
        for k, n in counts.items():
            marker = "" if n == len(manifest) else f"   <-- only {n}/{len(manifest)}"
            print(f"    {k:<24} {n:>3}{marker}")
        missing = [m["image_name"] for m in manifest if not m["capture_paths"]]
        if missing:
            print(f"    NO CAPTURES AT ALL for {len(missing)}: {missing}")

    return manifest

def evaluate_camera_capture_once(
    image_path,
    target_eval_np,
    out_dir,
    device,
    roi,
    out_hw,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
    tag="camera_capture",
):
    ensure_dir(out_dir)
    print("Saving outputs to:", out_dir)  # temporary debug
    
    t0 = time.time()

    if use_dc_exclusion:
        I_cam, cam_resized_np, dc_mask_t, dc_center_used = load_camera_capture_for_citl(
            image_path=image_path,
            roi=roi,
            out_hw=out_hw,
            device=device,
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
        )
    else:
        I_cam, cam_resized_np = no_dc_load_camera_capture_for_citl(
            image_path=image_path,
            roi=roi,
            out_hw=out_hw,
            device=device,
            angle=angle,
            eps_norm=eps_norm,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
        )
        dc_mask_t = None
        dc_center_used = None

    elapsed = time.time() - t0

    recon_np = I_cam[0].detach().cpu().numpy().astype(np.float32)
    metrics = compute_metrics(recon_np, target_eval_np)

    result = {
        "image_path": image_path,
        # "wall_clock_seconds": float(elapsed),
        "use_dc_exclusion": bool(use_dc_exclusion),
        **metrics,
    }

    if dc_center_used is not None:
        result["dc_center_used"] = [int(dc_center_used[0]), int(dc_center_used[1])]
        result["dc_radius"] = int(dc_radius)

    np.save(os.path.join(out_dir, f"{tag}_recon.npy"), recon_np)
    save_display_image(recon_np, os.path.join(out_dir, f"{tag}_recon.png"))

    np.save(os.path.join(out_dir, f"{tag}_raw_resized.npy"), cam_resized_np)
    save_display_image(cam_resized_np, os.path.join(out_dir, f"{tag}_raw_resized.png"))

    if dc_mask_t is not None:
        dc_mask_np = (dc_mask_t.detach().cpu().numpy().astype(np.uint8) * 255)
        Image.fromarray(dc_mask_np).save(os.path.join(out_dir, f"{tag}_dc_mask.png"))

    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result

def evaluate_single_target_experimental(
    target_image_path,
    image_name,
    capture_paths,
    out_dir,
    device,
    roi,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
):
    ensure_dir(out_dir)

    target = prepare_target_arrays(image_path=target_image_path, device=device)
    target_eval_np = target["target_eval_np"]
    target_amplitude_np = target["target_amplitude_np"]
    H, W = target["H"], target["W"]

    save_target_artifacts(out_dir, target_eval_np, target_amplitude_np)

    summary = {
        "image_name": image_name,
        "image_path": target_image_path,
        "summary": {
            "image_path": target_image_path,
            "slm_shape": [H, W],
            "comparison_mode": "experimental_multilevel",
            "regime1": {}
        }
    }

    # GS experimental (fixed chosen setting: 2000 iters)
    if "gs" in capture_paths:
        gs_result = evaluate_camera_capture_once(
            image_path=capture_paths["gs"],
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, "gs"),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag="gs_exp",
        )
        summary["summary"]["regime1"]["gs"] = {
            "iterations": 750,
            "psnr": gs_result["psnr"],
            "ssim": gs_result["ssim"],
            "mse": gs_result["mse"],
            "nmse": gs_result["nmse"],
            "diffraction_efficiency": gs_result["diffraction_efficiency"],
        }

    # GD experimental (fixed chosen setting: 20000 iters)
    if "gd" in capture_paths:
        gd_result = evaluate_camera_capture_once(
            image_path=capture_paths["gd"],
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, "gd"),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag="gd_exp",
        )
        summary["summary"]["regime1"]["gd"] = {
            "iterations": 750,
            "psnr": gd_result["psnr"],
            "ssim": gd_result["ssim"],
            "mse": gd_result["mse"],
            "nmse": gd_result["nmse"],
            "diffraction_efficiency": gd_result["diffraction_efficiency"],
        }

    # Per-image transformer experimental
    if "transformer" in capture_paths:
        tf_result = evaluate_camera_capture_once(
            image_path=capture_paths["transformer"],
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, "transformer"),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag="transformer_exp",
        )
        summary["summary"]["regime1"]["transformer"] = {
            "psnr": tf_result["psnr"],
            "ssim": tf_result["ssim"],
            "mse": tf_result["mse"],
            "nmse": tf_result["nmse"],
            "diffraction_efficiency": tf_result["diffraction_efficiency"],
        }

    # Shared model experimental
    if "shared_model" in capture_paths:
        shared_result = evaluate_camera_capture_once(
            image_path=capture_paths["shared_model"],
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, "shared_model"),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag="shared_model_exp",
        )
        summary["summary"]["regime1"]["shared_model"] = {
            "psnr": shared_result["psnr"],
            "ssim": shared_result["ssim"],
            "mse": shared_result["mse"],
            "nmse": shared_result["nmse"],
            "diffraction_efficiency": shared_result["diffraction_efficiency"],
        }

    known_methods = {"gs", "gd", "transformer", "shared_model"}

    for method_name, capture_path in capture_paths.items():
        if method_name in known_methods:
            continue

        result = evaluate_camera_capture_once(
            image_path=capture_path,
            target_eval_np=target_eval_np,
            out_dir=os.path.join(out_dir, method_name),
            device=device,
            roi=roi,
            out_hw=(H, W),
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
            tag=method_name,
        )

        summary["summary"]["regime1"][method_name] = {
            "psnr": result["psnr"],
            "ssim": result["ssim"],
            "mse": result["mse"],
            "nmse": result["nmse"],
            "diffraction_efficiency": result["diffraction_efficiency"],
        }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary

def evaluate_experimental_dataset(
    dataset_manifest,
    out_dir,
    device,
    roi,
    angle=0,
    eps_norm=1e-12,
    dc_radius=45,
    auto_center=True,
    dc_center=None,
    subtract_min=True,
    median_ksize=0,
    use_dc_exclusion=True,
):
    ensure_dir(out_dir)

    all_results = []

    for item in dataset_manifest:
        image_name = item["image_name"]
        target_image_path = item["image_path"]
        capture_paths = item["capture_paths"]

        print(f"\n=== Evaluating {image_name} ===")

        target_out_dir = os.path.join(out_dir, image_name)

        result = evaluate_single_target_experimental(
            target_image_path=target_image_path,
            image_name=image_name,
            capture_paths=capture_paths,
            out_dir=target_out_dir,
            device=device,
            roi=roi,
            angle=angle,
            eps_norm=eps_norm,
            dc_radius=dc_radius,
            auto_center=auto_center,
            dc_center=dc_center,
            subtract_min=subtract_min,
            median_ksize=median_ksize,
            use_dc_exclusion=use_dc_exclusion,
        )

        all_results.append(result)

    with open(os.path.join(out_dir, "comparison_experimental_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
