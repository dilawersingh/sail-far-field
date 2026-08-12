r"""
paths.py - where everything lives, resolved from this file's own location.

WHY THIS EXISTS.

  This deposit is self-contained: the targets, the results and the configuration
  are all inside it. So nothing should need editing before it runs. Every path is
  resolved relative to the deposit root, which is found from this file's location
  rather than from the working directory - so it behaves the same whether you run
  a script from the repository root, from code/analysis/, or from a notebook
  opened somewhere else entirely.

  Set SAIL_ROOT to point the whole thing at a different tree (for example, at
  your own results while re-running the experiments). Everything else follows.

USAGE
    import paths
    print(paths.TARGETS)          # <deposit>/targets
    print(paths.RESULTS)          # <deposit>/results
    print(paths.SCORED)           # <deposit>/output/sail_scored.json
    paths.report()                # print every location and whether it exists
"""
from __future__ import annotations

import os
from pathlib import Path

# code/analysis/paths.py -> code/analysis -> code -> <deposit root>
_HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("SAIL_ROOT", _HERE.parents[2]))

TARGETS = ROOT / "targets"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "code" / "configurations"

# roi.json is an INPUT - the magnified-region coordinates, chosen by hand from
# the targets before any reconstruction was viewed. It is versioned with the
# analysis code because the figures read it; it is not produced by anything
# here.
ROI = _HERE.parent / "roi.json"

# The results tree mirrors the three recorded roots the analysis reads. The
# scorer's environment overrides (SAILREV_SA and friends) derive from these
# in the notebook bootstrap, so no module carries a path of its own.
SA = RESULTS / "Self-Attention" / "multilevel"
GDCITL = RESULTS / "GD+CITL"
FNO = RESULTS / "FNO"

EXPERIMENTS = SA / "experiments"
SIMULATIONS = SA / "simulations"
# The matched-compute sweep trees. The 750-iteration operating point reported
# in the paper is read off these same runs; there is exactly one GS/GD
# simulation dataset.
SIM_SWEEP_IDEAL = SIMULATIONS / "simulation_comparison_ideal_10k"
SIM_SWEEP_FAITHFUL = SIMULATIONS / "simulation_comparison_faithful_10k"
REPLAY = EXPERIMENTS / "replay_converged"
ABERRATION = EXPERIMENTS / "aberration_sweep"

# WHAT THE ANALYSIS PRODUCES, all under one root.
#
# results/ holds what the EXPERIMENTS produced - captures, run logs, simulation
# sweeps. output/ holds what the ANALYSIS produced from them. Keeping the two
# apart means a rebuild can never write into the deposit's own record of what
# was run.
OUTPUT = ROOT / "output"
SCORED = OUTPUT / "sail_scored.json"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"


def resolve(value):
    """Turn a config path value into an absolute path under ROOT.

    Accepts a string or the single-element list the configs use. Relative values
    are taken as relative to the deposit root; absolute values are left alone, so
    pointing a config at an external results tree still works.
    """
    if isinstance(value, (list, tuple)):
        value = value[0]
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)


def report() -> None:
    print(f"  ROOT      {ROOT}"
          f"{'' if ROOT.exists() else '   ** MISSING **'}")
    for name in ("TARGETS", "RESULTS", "CONFIGS", "ROI",
                 "SA", "GDCITL", "FNO",
                 "EXPERIMENTS", "SIMULATIONS", "SIM_SWEEP_IDEAL",
                 "SIM_SWEEP_FAITHFUL", "REPLAY", "ABERRATION",
                 "OUTPUT", "SCORED", "FIGURES", "TABLES"):
        p = globals()[name]
        mark = "" if p.exists() else "   ** missing **"
        print(f"  {name:18s} {p}{mark}")


if __name__ == "__main__":
    report()
