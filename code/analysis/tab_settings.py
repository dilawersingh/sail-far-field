r"""
tab_settings.py -- T3. Implementation settings, read from the runs' own
run_configuration.json files, never asserted.

Full implementation detail, read from the archived configs, and the
verification of tab_results.CONSTANTS against them, executed on every
notebook rebuild.

METHOD. Every run in this study archived a config snapshot
(run_configuration.json) at launch. T3 does not restate settings from memory:
for each method family it globs ALL of that family's config snapshots across
both physics conditions, extracts the reported fields, and REQUIRES them to
agree across every run in the family. A field that differs between runs is
rendered as MIXED(...) in the table, which cannot survive proofreading, so a
drifted run can never hide behind a clean-looking table. The printed n per
family says how many snapshots agree.

Five products:
  T3a (t3_settings.tex)  : learned-model families x training/architecture.
  T3b (t3_baselines.tex) : classical GS/GD+CITL settings.
  T3c (t3_hardware.tex)  : shared hardware and the two physics conditions,
                           from the CITL configs' conditions block.
  T3d (t3_fno.tex)       : FNO baseline settings (full-coverage w8/m500 run;
                           standard unmodified FNO only, per the B5 scope).
  T3e (t3_regime_map.tex): the training-regime map. Several reviewer
                           comments (R1.5, R2.6, R3.4) read two different
                           training regimes as one; this is the single place
                           a reader resolves any regime name to what
                           supervises it, how many targets share one model,
                           and where it is reported. Budgets are read from
                           the same config snapshots; only the structural
                           columns are curated text.

Also cross-checks tab_results.CONSTANTS entries that are derivable from
configs (sim_epochs, sail_epochs, gd_iterations, gs_iterations_bench) and
prints OK or MISMATCH for each.

SCOPE. sail+ and sail_no_dc runs are deliberately excluded: neither is
reported in the revision (sail+ removed from the manuscript; sail_no_dc not
reported). FNO is the standard unmodified network only (fno_scratch,
fno_regress).

Usage (notebook):
    import tab_settings
    tab_settings.build(OUT / "tables")
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import sailrev as S

MIXED = "MIXED"


def _root() -> Path:
    return Path(os.environ["SAILREV_RESULTS"])


# --------------------------------------------------------------------------
# Config discovery: family -> list of run_configuration.json paths
# --------------------------------------------------------------------------
def family_globs() -> dict[str, list[str]]:
    """Glob patterns relative to SAILREV_RESULTS, both physics included."""
    sa = "Self-Attention/multilevel"
    return {
        "transformer_per_target":
            [f"{sa}/simulations/per_target/1000px_P500_B1_*/*/run_configuration.json"],
        "transformer_batched":
            [f"{sa}/simulations/batched/1000px_P500_B18_*/run_configuration.json"],
        "sail":
            [f"{sa}/experiments/sail/*/*/run_configuration.json"],
        "batched_sail_750":
            [f"{sa}/experiments/batched_sail/*/*/run_configuration.json"],
        "batched_sail_2000":
            [f"{sa}/experiments/batched_sail_2k/*/*/run_configuration.json"],
        "gd_citl":
            ["GD+CITL/experiments/*/citl_gs_gd_*/*/run_configuration.json"],
        "fno":
            ["FNO/rebuttal/fno_rebuttal_*/run_configuration.json"],
    }


def load_family(name: str) -> tuple[dict, int]:
    """(merged config fields, n snapshots). Fields where snapshots disagree
    come back as 'MIXED(v1 | v2)'. Only leaves under the keys we report are
    compared, so incidental per-run fields (paths, timestamps) never trip it."""
    root = _root()
    paths: list[Path] = []
    for g in family_globs()[name]:
        paths += sorted(root.glob(g))
    paths = [p for p in paths if "_rig_calibration" not in str(p)]
    if not paths:
        raise FileNotFoundError(f"T3: no run_configuration.json for {name}")
    merged: dict = {}
    for p in paths:
        cfg = json.loads(p.read_text())
        flat = _flatten(cfg)
        for k, v in flat.items():
            if k not in merged:
                merged[k] = v
            elif merged[k] != v and not str(merged[k]).startswith(MIXED):
                merged[k] = f"{MIXED}({merged[k]} | {v})"
    return merged, len(paths)


def _flatten(cfg: dict) -> dict:
    """Pull out the comparable, reportable leaves; ignore paths/experiment
    metadata. Handles the key spellings that vary across the codebase's eras
    (global_patch vs coarse_patch; physics in 'conditions', 'physics', or
    inline in hyperparameters)."""
    out = {}
    hp = cfg.get("hyperparameters", {})
    for k in ("epochs", "batch_size", "learning_rate", "heads", "layers",
              "embedding_dimension", "feed_forward_dim", "dropout",
              "pre_norm", "height", "width"):
        if k in hp:
            out[k] = hp[k]
    if "coarse_patch" in hp:
        out["patch"] = hp["coarse_patch"]
    if "global_patch" in hp:
        out["patch"] = hp["global_patch"]
    run = cfg.get("run", {})
    for k in ("height", "width", "lambda_sim", "gs_iterations",
              "gd_iterations", "gd_lr", "gd_optimizer_name", "gd_tv_weight",
              "target_formulation", "calib_burnin_iterations"):
        if k in run:
            out[k] = run[k]
    model = cfg.get("model", {})
    for k in ("modes1", "modes2", "width", "num_layers", "projection_width",
              "activation", "domain_padding"):
        if k in model:
            out[f"fno.{k}"] = model[k]
    hp2 = cfg.get("hyperparameters", {})
    for k in ("epochs_scratch", "epochs_regress", "weight_decay",
              "lr_decay_every", "lr_decay_factor"):
        if k in hp2:
            out[k] = hp2[k]
    hw = cfg.get("hardware", {})
    for k in ("slm_shape", "dc_radius", "dc_center", "settle_time_s",
              "calibration_mode"):
        if k in hw:
            out[k] = hw[k]
    if "manual_alignment" in hw:
        out["alignment"] = hw["manual_alignment"]
    # physics: named conditions (CITL-era), a 'physics' block (batched sim),
    # or inline hyperparameters (per-target sim)
    conds = {}
    for c in cfg.get("conditions", []):
        conds[c["physics_name"]] = {k: c[k] for k in
                                    ("pad_factor", "apply_sinc", "fill_factor")}
    if "physics" in cfg:
        ph = cfg["physics"]
        name = "ideal" if not ph.get("apply_sinc") else "faithful"
        conds[name] = {k: ph[k] for k in
                       ("pad_factor", "apply_sinc", "fill_factor")}
    if "pad_factor" in hp:
        name = "ideal" if not hp.get("apply_sinc") else "faithful"
        conds[name] = {k: hp[k] for k in
                       ("pad_factor", "apply_sinc", "fill_factor")}
    for name, c in conds.items():
        out[f"physics.{name}"] = c
    return out


# --------------------------------------------------------------------------
# T3a: learned models
# --------------------------------------------------------------------------
T3A_FAMILIES = ["transformer_per_target", "transformer_batched", "sail",
                "batched_sail_750", "batched_sail_2000"]
T3A_LABELS = {"transformer_per_target": "Transformer, per-target (sim.)",
              "transformer_batched": "Transformer, batched (sim.)",
              "sail": "SAIL (per-target)",
              "batched_sail_750": "Batched SAIL (750)",
              "batched_sail_2000": "Batched SAIL (2000)"}
# (row label, config key, formatter) -- STRUCTURAL rows (camera feedback) are
# implied by the family definition, not read; they are marked below.

# --------------------------------------------------------------------------
# HARDCODED IN THE TRAINING SCRIPTS, NOT IN ANY CONFIG (2026-08-10).
# The learned families' run_configuration.json snapshots do not record the
# optimizer's weight decay or its learning-rate schedule, because both are
# literals in the training loop rather than config fields. They are still
# settings a reader needs, so they are declared here WITH THEIR SOURCE, the
# same convention tab_results.CONSTANTS uses for the parameter counts.
# VERIFY against the script before submission; these are the only values in
# this module that are asserted rather than read.
# --------------------------------------------------------------------------
SCRIPT_CONSTANTS = {
    "sail": {
        "lr_schedule": ("$\\times$0.95 every 250 epochs",
                        "sail_citl_transformer.py training loop"),
        "weight_decay": ("1e-4", "sail_citl_transformer.py, AdamW(...)"),
        "optimiser": ("AdamW", "training loop, torch.optim.AdamW(...)"),
        "betas": ("(0.9, 0.999)", "training loop, torch.optim.AdamW(...)"),
    },
    "batched_sail_750": {
        "lr_schedule": ("$\\times$0.95 every 250 epochs",
                        "batched SAIL training loop, shared with per-target"),
        "weight_decay": ("1e-4", "batched SAIL, AdamW(...)"),
        "optimiser": ("AdamW", "training loop, torch.optim.AdamW(...)"),
        "betas": ("(0.9, 0.999)", "training loop, torch.optim.AdamW(...)"),
    },
    "batched_sail_2000": {
        "lr_schedule": ("$\\times$0.95 every 250 epochs",
                        "batched SAIL training loop, shared with per-target"),
        "weight_decay": ("1e-4", "batched SAIL, AdamW(...)"),
        "optimiser": ("AdamW", "training loop, torch.optim.AdamW(...)"),
        "betas": ("(0.9, 0.999)", "training loop, torch.optim.AdamW(...)"),
    },
    # THE PERIOD DIFFERS BETWEEN SIMULATION AND BENCH, and it is not an
    # oversight (Dilawer 2026-08-10). Simulation trains for 10,000 epochs and
    # decays every 1,000; the camera-in-the-loop runs train for 750 and decay
    # every 250, because a 1,000-epoch period would leave a 750-epoch run at
    # its initial rate for the whole run. Same factor throughout.
    # Corroborated by fno_simulation.py, which mirrors the simulation loop and
    # whose config records 0.95 every 1,000.
    "transformer_per_target": {
        "lr_schedule": ("$\\times$0.95 every 1000 epochs",
                        "sail.ipynb simulation loop; 10,000-epoch budget"),
        "weight_decay": ("1e-4", "sail.ipynb, AdamW(...)"),
        "optimiser": ("AdamW", "training loop, torch.optim.AdamW(...)"),
        "betas": ("(0.9, 0.999)", "training loop, torch.optim.AdamW(...)"),
    },
    "transformer_batched": {
        "lr_schedule": ("$\\times$0.95 every 1000 epochs",
                        "sail.ipynb simulation loop; 10,000-epoch budget"),
        "weight_decay": ("1e-4", "sail.ipynb, AdamW(...)"),
        "optimiser": ("AdamW", "training loop, torch.optim.AdamW(...)"),
        "betas": ("(0.9, 0.999)", "training loop, torch.optim.AdamW(...)"),
    },
}

T3A_ROWS = [
    ("Epochs", "epochs", str),
    ("Batch size", "batch_size", str),
    ("Learning rate", "learning_rate", lambda v: f"{v:g}"),
    ("Patch size $p$", "patch", str),
    ("Heads", "heads", str),
    ("Encoder layers", "layers", str),
    ("$d_\\mathrm{model}$", "embedding_dimension", str),
    ("$d_\\mathrm{ff}$", "feed_forward_dim", str),
    ("Dropout", "dropout", lambda v: f"{v:g}"),
    ("Pre-norm", "pre_norm", lambda v: "yes" if v is True else str(v)),
    ("$\\lambda_\\mathrm{sim}$", "lambda_sim", lambda v: f"{v:g}"),
    # The decay is real and was invisible outside Algorithm 1 until 2026-08-09.
    # Read from the config rather than asserted, and "--" means the config did
    # not record one, not that the rate was held flat.
    ("Optimiser", "__optimiser__", None),
    ("Betas", "__betas__", None),
    ("Weight decay", "__weight_decay__", None),
    ("LR schedule", "__lr_schedule__", None),
    ("Grid", None, None),          # height x width, composed
]
CAMERA_FEEDBACK = {"transformer_per_target": "no", "transformer_batched": "no",
                   "sail": "every epoch", "batched_sail_750": "every epoch",
                   "batched_sail_2000": "every epoch"}

# T3e structural columns:
#   (family, label, supervision, targets/model, models, reported in, budget)
# budget=None means "look the epoch count up in the run configuration"; a
# string is a literal, used for the iterative baselines, which carry iteration
# budgets rather than epochs and have no configuration family of their own.
#
# EVERY METHOD IS LISTED, not only the learned ones (Dilawer 2026-08-09). This
# table is the reader's index from method to figure, so a method missing from
# it is a method they cannot locate.
# "Reported in" is manuscript-facing and was last correct when batched
# SAIL was Fig 4. Corrected 2026-08-09 to the submitted numbering: Fig 4
# is the qualitative figure, Fig 5 is batched SAIL. SAIL is not in E5
# (that figure is GS/GD against a dashed transformer level), E3c is named
# alongside E3a, and T1/T2 are spelled as the manuscript spells them.
REGIME_MAP = [
    ("transformer_per_target", "Transformer, per-target (simulation)",
     "simulated forward model", "1", "18 per physics",
     "Fig 2, Fig 3, Fig 4, E3a, E3c, E5, E8, Table 1", None),
    ("transformer_batched", "Transformer, batched (simulation)",
     "simulated forward model", "18", "1 per physics",
     "Fig 2, Fig 3, E3a, E3c, Table 1", None),
    ("sail", "SAIL (per-target)",
     "physical optics, every epoch", "1", "18 per physics",
     "Fig 3, Fig 4, Fig 5, E3a, E3c, E8, Table 1, Table 2", None),
    ("batched_sail_750", "Batched SAIL, 750 epochs",
     "physical optics, every epoch", "18", "1 per physics",
     "Fig 5, Table 1, Table 2", None),
    ("batched_sail_2000", "Batched SAIL, 2000 epochs",
     "physical optics, every epoch", "18", "1 per physics",
     "Fig 4, Fig 5, E3a, E3c, E8, Table 1, Table 2", None),
    ("gs", "GS", "none (no training)", "--", "none",
     "Fig 2, Fig 3, Fig 4, E4, E5, E8, Table 1", "750 iterations"),
    ("gs_10000", "GS, converged", "none (no training)", "--", "none",
     "E5, Table 1", "10,000 iterations"),
    ("gd", "GD", "none (no training)", "--", "none",
     "Fig 2, Fig 3, Fig 4, E4, E5, E8, Table 1", "750 iterations"),
    ("gd_10000", "GD, converged", "none (no training)", "--", "none",
     "E5, Table 1", "10,000 iterations"),
    ("gs_citl_random", "GS+CITL, random init",
     "physical optics, every iteration", "--", "none", "Fig 3, Table 1",
     "25 iterations"),
    ("gs_citl_warm", "GS+CITL, simulation-seeded",
     "physical optics, every iteration", "--", "none", "Fig 3, Table 1",
     "25 iterations"),
    ("gd_citl_random", "GD+CITL, random init",
     "physical optics, every iteration", "--", "none",
     "Fig 3, Table 1, Table 2", "750 iterations"),
    ("gd_citl_warm", "GD+CITL, simulation-seeded",
     "physical optics, every iteration", "--", "none",
     "Fig 3, Fig 4, E8, Table 1, Table 2", "750 iterations"),
]


def _cell(fam_cfg: dict, key, fmt) -> str:
    if key is None:                      # composed Grid row
        h, w = fam_cfg.get("height"), fam_cfg.get("width")
        return f"{h}$\\times${w}" if h and w else "--"
    if key == "__lr_schedule__":         # composed LR-decay row
        every = fam_cfg.get("lr_decay_every")
        factor = fam_cfg.get("lr_decay_factor")
        if every is None or factor is None:
            return "--"
        return f"$\\times${factor:g} every {every} epochs"
    if key == "__script_const__":
        return "--"                      # replaced per family in build()
    v = fam_cfg.get(key)
    if v is None:
        return "--"
    if isinstance(v, str) and v.startswith(MIXED):
        return v
    return fmt(v)


def build(out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = {}
    for f in T3A_FAMILIES + ["gd_citl", "fno"]:
        cfg, n = load_family(f)
        fams[f] = cfg
        mixed = [k for k, v in cfg.items()
                 if isinstance(v, str) and v.startswith(MIXED)]
        flag = f"  !! MIXED fields: {mixed}" if mixed else ""
        print(f"T3 | {f:24s} n={n:3d} config snapshots agree{flag}")

    # ---- T3a ----
    print("\nT3a | learned-network settings (from run_configuration.json)")
    L = [r"\begin{tabular}{l" + "c" * len(T3A_FAMILIES) + "}", r"\toprule",
         "Setting & " + " & ".join(T3A_LABELS[f] for f in T3A_FAMILIES)
         + r" \\", r"\midrule"]
    hdr = f"{'setting':22s}" + "".join(f"{T3A_LABELS[f]:>28s}"
                                       for f in T3A_FAMILIES)
    print(hdr)
    for label, key, fmt in T3A_ROWS:
        cells = [_cell(fams[f], key, fmt) for f in T3A_FAMILIES]
        # script-sourced values win over a missing config field, never over a
        # present one, so the config stays the authority wherever it speaks
        const_key = {"__lr_schedule__": "lr_schedule",
                     "__weight_decay__": "weight_decay",
                     "__optimiser__": "optimiser",
                     "__betas__": "betas"}.get(key)
        if const_key:
            cells = [SCRIPT_CONSTANTS.get(f, {}).get(const_key, ("--",))[0]
                     if c == "--" else c
                     for c, f in zip(cells, T3A_FAMILIES)]
        print(f"{label:22s}" + "".join(f"{c:>28s}" for c in cells))
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L.append("Camera feedback & " +
             " & ".join(CAMERA_FEEDBACK[f] for f in T3A_FAMILIES) + r" \\")
    print(f"{'Camera feedback':22s}" +
          "".join(f"{CAMERA_FEEDBACK[f]:>28s}" for f in T3A_FAMILIES))
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "t3_settings.tex").write_text("\n".join(L))
    print(f"  table -> {out_dir / 't3_settings.tex'}")

    # ---- T3b: classical baselines ----
    g = fams["gd_citl"]
    rows_b = [
        ("GS iterations (camera feedback)", _cell(g, "gs_iterations", str)),
        ("GD iterations", _cell(g, "gd_iterations", str)),
        ("GD optimiser", _cell(g, "gd_optimizer_name", str)),
        ("GD learning rate", _cell(g, "gd_lr", lambda v: f"{v:g}")),
        ("GD TV weight", _cell(g, "gd_tv_weight", lambda v: f"{v:g}")),
        ("Target formulation", _cell(g, "target_formulation", str)),
        ("Grid", _cell(g, None, None)),
        ("Init arms", "random, simulation-seeded"),
    ]
    # PER METHOD, NOT PER SETTING (Dilawer 2026-08-09). The old shape listed
    # "GS iterations (camera feedback) 25" and "GD iterations 750" side by
    # side, which left a reader unable to tell which number belonged to the
    # plain baseline and which to its CITL arm, and hid the fact that only the
    # CITL arms have two initializations. Config-sourced cells are marked in
    # the caption; the converged 10,000-iteration arm and the alternating-
    # projection entries are structural, not configurable.
    gs_it = _cell(g, "gs_iterations", str)
    gd_it = _cell(g, "gd_iterations", str)
    gd_opt = _cell(g, "gd_optimizer_name", str)
    gd_lr = _cell(g, "gd_lr", lambda v: f"{v:g}")
    rows_b2 = [
        ("GS", f"{gd_it} and 10,000", "alternating projection", "--",
         "random", "none"),
        ("GD", f"{gd_it} and 10,000", gd_opt, gd_lr, "random", "none"),
        ("GS+CITL", gs_it, "alternating projection", "--",
         "random, simulation-seeded", "every iteration"),
        ("GD+CITL", gd_it, gd_opt, gd_lr,
         "random, simulation-seeded", "every iteration"),
    ]
    print("\nT3b | classical baselines, per method")
    print(f"  {'method':10s} {'iterations':18s} {'optimiser':24s} "
          f"{'lr':>6s}  {'initialization':26s} camera")
    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Method & Iterations & Optimiser & LR & Initialization & Camera \\",
         r"\midrule"]
    for m, it, opt, lr, init, cam in rows_b2:
        print(f"  {m:10s} {it:18s} {opt:24s} {lr:>6s}  {init:26s} {cam}")
        L.append(f"{m} & {it} & {opt} & {lr} & {init} & {cam} \\\\")
    L.append(r"\midrule")
    for label, v in rows_b:
        if label in ("Target formulation", "Grid", "GD TV weight"):
            print(f"  {label:34s} {v}")
            L.append(f"\\multicolumn{{6}}{{l}}{{{label}: {v}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "t3_baselines.tex").write_text("\n".join(L))
    print(f"  table -> {out_dir / 't3_baselines.tex'}")

    # ---- T3c: hardware + physics conditions ----
    slm = g.get("slm_shape"); dc = g.get("dc_center")
    al = g.get("alignment", {})
    rows_c = [
        ("SLM", f"{slm[0]}$\\times${slm[1]}" if isinstance(slm, list) else "--"),
        ("DC block centre (px)", f"({dc[0]}, {dc[1]})"
         if isinstance(dc, list) else "--"),
        ("DC block radius (px)", _cell(g, "dc_radius", str)),
        ("SLM settle time (s)", _cell(g, "settle_time_s", lambda v: f"{v:g}")),
        ("Calibration", _cell(g, "calibration_mode", str) +
         (f", rotation {al.get('rotation_deg')}$^\\circ$" if al else "")),
        ("Calibration burn-in (iter.)", _cell(g, "calib_burnin_iterations",
                                              str)),
    ]
    for phys in ("ideal", "faithful"):
        c = g.get(f"physics.{phys}")
        if isinstance(c, dict):
            rows_c.append(
                (f"{phys.capitalize()} forward model",
                 f"pad {c['pad_factor']}, sinc "
                 f"{'on' if c['apply_sinc'] else 'off'}, "
                 f"fill {c['fill_factor']:g}"))
    print("\nT3c | shared hardware and forward-model conditions")
    L = [r"\begin{tabular}{ll}", r"\toprule", r"Setting & Value \\",
         r"\midrule"]
    for label, v in rows_c:
        print(f"  {label:34s} {v}")
        L.append(f"{label} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "t3_hardware.tex").write_text("\n".join(L))
    print(f"  table -> {out_dir / 't3_hardware.tex'}")

    # ---- T3d: FNO baseline ----
    f = fams["fno"]
    modes = (f.get("fno.modes1"), f.get("fno.modes2"))
    rows_d = [
        ("Fourier modes", f"{modes[0]} $\\times$ {modes[1]} (full coverage)"
         if all(modes) else "--"),
        ("Width", _cell(f, "fno.width", str)),
        ("Layers", _cell(f, "fno.num_layers", str)),
        ("Projection width", _cell(f, "fno.projection_width", str)),
        ("Activation", _cell(f, "fno.activation", str)),
        ("Domain padding", _cell(f, "fno.domain_padding", str)),
        ("Epochs, from scratch", _cell(f, "epochs_scratch", str)),
        ("Epochs, regression", _cell(f, "epochs_regress", str)),
        ("Learning rate", _cell(f, "learning_rate", lambda v: f"{v:g}") +
         " (probed over 5 orders; quality insensitive)"),
        ("Weight decay", _cell(f, "weight_decay", lambda v: f"{v:g}")),
        ("LR schedule", f"$\\times${f.get('lr_decay_factor', '?')} every "
         f"{f.get('lr_decay_every', '?')} epochs"),
        ("Grid", _cell(f, None, None)),
        ("Physics", "ideal only (stated scope choice)"),
        ("Conditions", "fno\\_scratch, fno\\_regress (standard FNO only)"),
    ]
    print("\nT3d | FNO baseline settings (full-coverage w8/m500 run)")
    L = [r"\begin{tabular}{ll}", r"\toprule", r"Setting & Value \\",
         r"\midrule"]
    for label, v in rows_d:
        print(f"  {label:34s} {v}")
        L.append(f"{label} & {v} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "t3_fno.tex").write_text("\n".join(L))
    print(f"  table -> {out_dir / 't3_fno.tex'}")

    # ---- T3e: the training-regime map ----
    print("\nT3e | training regimes (budgets from run_configuration.json)")
    print(f"{'regime':38s} {'in the training loop':30s} {'budget':>18s} "
          f"{'tgt/net':>9s} {'networks':>15s}   reported in")
    L = [r"\begin{tabular}{llrrrl}", r"\toprule",
         r"Regime & In the training loop & Budget & Targets & Networks & "
         r"Reported in \\",
         r" & & & per network & produced & \\", r"\midrule"]
    for fam, label, sup, tpm, models, where, budget in REGIME_MAP:
        epochs = (budget if budget is not None
                  else fams.get(fam, {}).get("epochs", "--"))
        print(f"{label:38s} {sup:30s} {epochs!s:>18s} {tpm:>9s} "
              f"{models:>15s}   {where}")
        L.append(f"{label} & {sup} & {epochs} & {tpm} & {models} & "
                 f"{where} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "t3_regime_map.tex").write_text("\n".join(L))
    print(f"  table -> {out_dir / 't3_regime_map.tex'}")


    # ---- CONSTANTS cross-check (closes the open_items action) ----
    import tab_results
    checks = [
        ("sim_epochs", fams["transformer_per_target"].get("epochs")),
        ("sail_epochs", fams["sail"].get("epochs")),
        ("gd_iterations", fams["gd_citl"].get("gd_iterations")),
        ("gs_iterations_bench", fams["gd_citl"].get("gs_iterations")),
    ]
    print("\nT3 | tab_results.CONSTANTS vs run_configuration.json:")
    for name, measured in checks:
        asserted = tab_results.CONSTANTS[name][0]
        ok = "OK" if measured == asserted else "MISMATCH"
        print(f"  {name:22s} asserted {asserted!s:>8s}  "
              f"config {measured!s:>8s}  {ok}")
    return None
