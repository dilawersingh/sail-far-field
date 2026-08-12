r"""
figstyle.py -- the single place every visual decision for the revised
manuscript is made. Same role as gd_citl_analysis/figstyle.py in the GD+CITL
paper: every module that produces media imports this one, so a change here
propagates to every figure and none of them carries a local literal.

WHAT LIVES HERE
    font family and EVERY font size          (SIZE)
    the method colour system                 (COOL, WARM, METHOD_COLORS)
    the accent reserved for the headline     (ACCENT)
    panel labels, axis styling, saving       (helpers)

WHAT DOES NOT
    display names for methods, which are terminology rather than style and
    live in sailrev.LABEL; and anything that reads data.

RULE: a figure module must never write a raw hex or a raw fontsize. If a size
is missing from SIZE, add it here rather than inlining a number, or the next
change to the manuscript's typography has to be made in six files.

THE COLOUR SYSTEM
    Four hue families, one per algorithm lineage: TEAL for the GS lineage,
    RED for GD, PURPLE for the learned models without hardware adaptation,
    BLUE for the SAIL family. Lightness within a family tracks how much
    machinery the method has, so darker means camera feedback has been added.
    SAIL is the only blue on the page. Full rationale, measured separations and
    the one constraint that must survive editing are on the block above the
    definitions below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Typography -- family AND every size, in one place
# --------------------------------------------------------------------------
FONT = "Arial"          # the manuscript's face; figures must match it

SIZE = {"base":18,"title":14,"title_wide":15.5,"label":14,"tick":13,
        "tick_small":12,"panel":18,"annot":12.5,"annot_big":15.5,"note":14,"legend":13.5}
# Sizes bumped 2026-08-04 (Dilawer: publication readability pass). Every module
# inherits these; never set a fontsize literal in a figure module.

RC = {"font.family": FONT,
      "font.size": SIZE["base"],
      "axes.titlesize": SIZE["title"],
      "axes.labelsize": SIZE["label"],
      "xtick.labelsize": SIZE["tick"],
      "ytick.labelsize": SIZE["tick"]}

_FONT_WARNED = False


def font_available(name: str = FONT) -> bool:
    from matplotlib import font_manager
    return any(f.name == name for f in font_manager.fontManager.ttflist)


def apply_style(strict: bool = False):
    """Apply the manuscript style. Complains loudly if the font is missing.

    strict=True raises instead of warning; the deposit's rebuild uses it so a
    machine without Arial cannot silently emit figures in another face.
    """
    global _FONT_WARNED
    import matplotlib.pyplot as plt
    if not font_available():
        msg = (f"{FONT} is not installed: figures will render in a fallback "
               f"face and will NOT match the manuscript. Install {FONT}, or "
               f"set figstyle.RC['font.family'] deliberately for a draft.")
        if strict:
            raise RuntimeError(msg)
        if not _FONT_WARNED:
            print(f"  WARNING: {msg}")
            _FONT_WARNED = True
    plt.rcParams.update(RC)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------
# FOUR HUE FAMILIES, one per algorithm lineage, lightness within each family
# tracking how much machinery the method has. This is the system:
#
#   TEAL    GS lineage          light -> dark as camera feedback is added
#   RED     GD lineage          light -> dark as camera feedback is added
#   PURPLE  transformer, no hardware adaptation (per-target, batched)
#   BLUE    SAIL family, the paper's method (per-target, batched)
#
# A reader sees four hues, one per lineage, and reads darkness within a hue as
# "more machinery". SAIL is the only blue on the page, which is what makes the
# comparison figures legible at a glance.
#
# MEASURED (dataviz validator, light surface, OKLab x100), interleaved worst
# cases: teal vs red 9.3 CVD / 27.1 normal; teal vs blue 15.2 / 15.8; red vs
# purple 18.2 / 18.9; purple vs blue 8.7 / 18.3. All clear.
#
# WHY TEAL AND NOT GREEN, AND WHY RED AND NOT ORANGE. The GD lineage wants a
# red family: light orange and yellow are the hardest colours to see on white
# (the old #ffb74d had 1.69:1 contrast). But red against green is the classic
# deuteranopia collision and measures dE 1.2 -- indistinguishable. Pink against
# green is barely better at 3.9. The fix is to move the OTHER family: teal
# against red measures 9.3, so the GS lineage went teal and GD went red. Do not
# reintroduce green here while GD is red.
#
# RED FAMILY SPACING. The two CITL arms (random init and simulation-seeded)
# sat only dE ~9 apart at #d32f2f / #b71c1c and read as the same colour in a
# figure. The darkest step went to #7f1010 to open that gap. The LIGHT step
# stays at #ef5350 and must not be lightened: at #f4756f it collides with the
# mid teal under protanopia (dE 3.8). So the family widens downwards only.
#
# The darkest teal sits just under the chroma floor (0.086, reads slightly
# grey). Accepted: it is the darkest step of a family whose lighter steps carry
# ample chroma, and it is always shown beside its own family.
# The lightest steps sit below 3:1 contrast on white. Acceptable ONLY because
# every figure using them carries direct labels and T1 is the table view.
TEAL = ["#14b8a6", "#0d9488", "#0e7a70"]      # GS, GS+CITL random, seeded
GREEN = TEAL                                  # alias, older figure code
RED = ["#ef5350", "#c62828", "#7f1010"]       # GD, GD+CITL random, seeded
ORANGE_F = RED                                # alias, older figure code
PURPLE_F = ["#ba68c8", "#6a1b9a"]             # transformer, batched transformer
BLUE = ["#1e88e5", "#5c9ce6", "#0d47a1"]      # SAIL, batched 750, batched 2000

# Reserved for the attention-selectivity result. It is not a method and must
# not borrow a method's colour: being off-palette marks it as a different KIND
# of quantity rather than another arm.
ACCENT = "#6836ff"
ACCENT_FAINT = "#b3a1ff"

COLOR_DIAG = "#8c8c8c"      # furniture only: zero lines, parity diagonals
COLOR_FAINT = "#bdbdbd"     # faint per-target traces behind a summary line

METHOD_COLORS = {
    # GS lineage, teal
    "gs": TEAL[0],
    "gs_citl_random": TEAL[1],
    "gs_citl_warm": TEAL[2],
    # GD lineage, red
    "gd": RED[0],
    "gd_citl_random": RED[1],
    "gd_citl_warm": RED[2],
    # simulation-only target-formulation arms: tints of their parent
    "gs_intensity": "#5eead4",
    "gd_intensity": "#f8a3a1",
    # learned, no hardware adaptation: purple
    "transformer_per_target": PURPLE_F[0],
    "transformer_batched": PURPLE_F[1],
    # SAIL family: blue, the only blue on the page
    "sail": BLUE[0],
    # 750 and 2000 MUST differ: Fig 4a exists to contrast them, and giving
    # both the same hex rendered that panel in one colour.
    "batched_sail_750": BLUE[1],
    "batched_sail_2000": BLUE[2],
    # FNO gets DELIBERATE NEUTRALS rather than a hue. It shares E1 with GD
    # (red) and the transformer (purple), so an earlier assignment out of the
    # purple family made the FNO-regression bar identical to the transformer
    # bar. Slate also reads correctly: FNO is the comparator architecture, not
    # one of the four lineages the hues encode. Low chroma is the trade, and it
    # is acceptable because these appear only in E1, where nothing else is
    # neutral. Separation against E1's other arms: 17.6 normal / 9.5 CVD.
    "fno_scratch": "#37474f",
    "fno_regress": "#90a4ae",
}

# The order methods are laid out in a figure. Structural (lineage, then
# adaptation, then ours), never by measured rank -- so the colour sequence runs
# smoothly down a y axis and a change in the data can never repaint a row.
CANONICAL_ORDER = ["gs", "gs_citl_random", "gs_citl_warm",
                   "gd", "gd_citl_random", "gd_citl_warm",
                   "transformer_per_target", "transformer_batched",
                   "sail", "batched_sail_750", "batched_sail_2000"]

# Backwards-compatible aliases for figure code written before this split.
COOL, WARM = GREEN + ORANGE_F, PURPLE_F + BLUE
PURPLE, ORANGE, PINK = PURPLE_F, ORANGE_F, PURPLE_F
COLOR_SAIL = METHOD_COLORS["sail"]
COLOR_BATCHED = METHOD_COLORS["batched_sail_2000"]
COLOR_TRANSFORMER = METHOD_COLORS["transformer_per_target"]
COLOR_GS = GREEN[0]
COLOR_GD = ORANGE_F[0]
COLOR_CITL = ORANGE_F[1]


def method_color(method: str) -> str:
    """The one place a method's colour is decided. Never inline a hex."""
    return METHOD_COLORS.get(method, COLOR_DIAG)


def is_ours(method: str) -> bool:
    return method in ("transformer_per_target", "transformer_batched", "sail",
                      "batched_sail_750", "batched_sail_2000")


def palette_table(label_fn=None) -> None:
    """Print method -> colour. The audit trail when a figure looks wrong."""
    lab = label_fn or (lambda m: m)
    groups = [("TEAL, GS lineage", ["gs", "gs_citl_random", "gs_citl_warm"]),
              ("RED, GD lineage", ["gd", "gd_citl_random", "gd_citl_warm"]),
              ("PURPLE, learned without hardware adaptation",
               ["transformer_per_target", "transformer_batched"]),
              ("BLUE, the SAIL family",
               ["sail", "batched_sail_750", "batched_sail_2000"]),
              ("other", ["gs_intensity", "gd_intensity",
                         "fno_scratch", "fno_regress"])]
    for title, ms in groups:
        print(title)
        for m in ms:
            print(f"  {METHOD_COLORS[m]}  {lab(m)}")
    print(f"accent (attention result): {ACCENT}, traces {ACCENT_FAINT}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def add_panel_label(ax, label_text, dx: float = -52, dy: float = 20):
    """Panel letter at a FIXED PIXEL offset from the axes' top-left corner.

    An axes-FRACTION offset lands somewhere different on every panel: a narrow
    panel's 18% is far fewer pixels than a wide one's, so "a" and "c" visibly
    failed to line up once (c) spanned the full figure width. Offset points are
    identical regardless of panel width, so panels sharing a left edge align.
    """
    ax.annotate(label_text, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=SIZE["panel"], fontweight="bold", va="top",
                ha="left", annotation_clip=False)


def halo(alpha: float = 0.82, pad: float = 1.6) -> dict:
    """bbox kwargs that put a soft white plate behind a text label.

    Numeric annotations sit on top of data by definition -- a median marker, a
    line, a swarm of points. Moving them elsewhere costs the association with
    what they label; a halo keeps them in place and legible over anything.
    Use for numbers ON the plot, never for axis labels or titles.
    """
    return {"facecolor": "white", "edgecolor": "none", "alpha": alpha,
            "pad": pad, "boxstyle": "round,pad=0.18"}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def padded_limits(x, y, frac=0.06):
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    span = (hi - lo) or 1.0
    pad = frac * span
    return lo - pad, hi + pad


def save(fig, out_path, dpi: int = 900):
    """Write PNG (for diffing) and PDF (for the manuscript).

    PNG carries no creation timestamp, so a byte-compare of PNGs answers "does
    it look the same"; two PDFs of an identical figure differ in bytes. Both
    are written for exactly that reason. Both now carry the same dpi, so the
    PDF is not a degraded copy of the PNG.

    WHY 900 AND NOT 600 (Dilawer 2026-08-07). A rasterised imshow panel is
    written at panel_width_inches * dpi pixels, so what matters is the panel's
    size on the page, not the size of the array behind it. The full-frame
    reconstructions are 1000 x 1000, and the smallest E8 grid panel is about
    1.22 in across, so 600 dpi produced 732 px and quietly downsampled the
    record. 900 dpi gives that same panel 1098 px, at or above native
    everywhere in the figure set, so nothing in the pipeline resamples the
    data downward. Purely vector figures are unaffected in content and cost
    almost nothing extra, because dpi only bites on rasterised content.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    # dpi matters for the PDF too. Vector elements are unaffected, but any
    # imshow panel is rasterised at this dpi when the PDF is written, and
    # without it the default figure dpi applies. That silently shrank the E8
    # reconstruction panels to 229 x 229 px in the PDF while the PNG of the
    # same figure carried them at full dpi (Dilawer 2026-08-07). The PDF now
    # keeps vector text AND full-resolution image data, so it is the better
    # source for every downstream document.
    fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    print(f"  figure -> {out_path.with_suffix('.png')}")
