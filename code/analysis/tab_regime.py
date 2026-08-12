r"""
tab_regime.py -- T4. Every cited comparator, located in its propagation
regime, with its bearing on our claim.

Nearly every learned-CGH comparator is Fresnel/near-field. This table is
what makes the near-field/far-field argument CHECKABLE instead of
assertable: each row is a specific citation, its regime as stated by its
own authors, and precisely what it does and does not imply for far-field
synthesis.

THE CLAIM, PRECISELY (caption material). Fresnel propagation retains partial
spatial locality in its transfer function, which is what makes convolutional
and local-window architectures viable there. Strict Fraunhofer propagation
is a Fourier transform: fully dense, every modulator pixel coupled to every
reconstruction point. Architectures and inductive biases developed for the
first regime have no established validity in the second; the cited
comparators are evidence for this premise, not against it.

THE CONCESSION THAT KEEPS IT CREDIBLE. The CITL algorithm itself is
regime-agnostic and transfers; we cite Peng et al. 2020 for our GD+CITL
baseline and dispute nothing about it. What does not transfer is the
architecture. The one prior far-field CITL instance found (Zimmermann et
al. 2025; Reichelt is last author) is calibration-only: the camera fits a
forward model once, then leaves the loop.

No data dependency: rows are curated citations, reviewed against the papers
themselves (see the skeleton's cross-cutting notes for the verification
trail). This module exists so T4 regenerates with the rest of the build and
so the citation list lives in exactly one place.

Usage (notebook):
    import tab_regime
    tab_regime.build(OUT / "tables")
"""
from __future__ import annotations

from pathlib import Path

# (work, regime, bearing)
ROWS = [
    ("Peng et al. 2020, ACM TOG 39(6) 185 (Neural Holography)",
     "Fresnel, 20 cm, near-eye",
     "Source of the CITL procedure; cited for our GD+CITL baseline, "
     "not disputed"),
    ("Peng et al. 2021, Sci. Adv. 7(46) eabg5040",
     "Fresnel, near-eye",
     "Partially coherent extension, same regime"),
    ("Choi et al. 2021, Optica 8(2) 143 (Michelson Holography)",
     "Fresnel",
     "Dual-SLM, same regime"),
    ("Chen et al. 2022, Opt. Lett. 47(4) 790",
     "Fresnel",
     "Off-axis CITL, same regime"),
    ("Chakravarthula et al. 2020, ACM TOG 39(6) 186",
     "Fresnel, near-eye",
     "Learned hardware-in-the-loop, same regime"),
    ("Liu et al. 2025, Nat. Commun. 16, 7761",
     "Fresnel",
     "Propagation-adaptive FNO across depth; a capability with no "
     "Fraunhofer analogue (no variable propagation distance to adapt to)"),
    ("Liu et al. 2025, Opt. Laser Technol. 181, 111740",
     "Fresnel",
     "Attention in the near-eye/CITL lineage, citing Chakravarthula"),
    ("Zimmermann et al. 2025, Opt. Eng. 64(9) 094102",
     "Fraunhofer",
     "The single far-field CITL instance found, and it is "
     "calibration-only: the camera fits a forward model once, then leaves "
     "the loop; no per-iteration feedback during generation"),
    ("This work (SAIL)",
     "Fraunhofer",
     "Per-iteration camera feedback inside far-field hologram generation"),
]


def build(out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("T4 | cited comparators by propagation regime")
    for work, regime, bearing in ROWS:
        print(f"  {work}\n    {regime:26s} {bearing}")
    L = [r"\begin{tabular}{p{0.34\linewidth}p{0.14\linewidth}"
         r"p{0.44\linewidth}}",
         r"\toprule", r"Work & Regime & Bearing on this study \\",
         r"\midrule"]
    for work, regime, bearing in ROWS:
        bold = r"\textbf{Fraunhofer}" if regime.startswith("Fraunhofer") \
            else regime
        L.append(f"{work} & {bold} & {bearing} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    out = out_dir / "t4_prior_art_regimes.tex"
    out.write_text("\n".join(L))
    print(f"  table -> {out}")
    print("T4 | strongest single sentence for the prose: the only prior "
          "far-field CITL work\nuses the camera to calibrate a forward model "
          "once, not to drive hologram generation,\nso per-iteration camera "
          "feedback for far-field synthesis has no prior instance that\nwe "
          "could find. Keep the concession beside it: the CITL algorithm "
          "transfers; the\narchitectures do not.")
    return None
