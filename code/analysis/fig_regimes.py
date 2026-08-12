r"""
fig_regimes.py -- the near-field/far-field regime comparison, visually.
One figure per target.

Fresnel propagation retains partial spatial locality, while strict
Fraunhofer propagation is a single Fourier transform, fully dense, every
modulator pixel coupled to every reconstruction point. This figure makes
that distinction visible. Two panels per target, everything else matched
(same grid, same GS algorithm, same iteration budget, zero-phase
initialisation; the build prints both regimes' reconstruction PSNR).

(a) The holograms. Phase-only hologram of the same target computed under
    Fresnel propagation (angular spectrum method, z = 10 cm) and under
    Fraunhofer propagation (Fourier transform). The near-field phase
    visibly retains the spatial layout of the scene; the far-field phase
    is structureless speckle at every scale.

(b) The locality probe. Add a constant pi offset to one small patch of
    each hologram, a localized modulator defect, and image |delta I| in
    the reconstruction. Near field, a compact disturbance at the
    corresponding location; far field, change across the entire
    reconstruction, because a hologram patch there is not a place in the
    image but a band of plane-wave components of the whole image.

The printed quantitative anchor is the fraction of |delta I| energy
within a box of 3 patch widths about the patch position, against the
uniform-spread expectation (the box's share of the field area). On
targets whose energy is itself concentrated at the patch position the
box metric is degenerate in both regimes.

The probe is a smooth pi patch rather than a randomized one: a
random-phase patch radiates into the modulator's full angular bandwidth
in any regime, so it is maximally delocalized by construction and probes
the pitch rather than the propagation physics. A constant offset has the
patch's own bandwidth.

Illustrative simulation, by design: pure numpy propagation, deterministic
(zero-phase init, no randomness), at the rig's modulator parameters
(PARAMS, printed by the build). Nothing here enters any quantitative
claim; the scored domains are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

import sailrev as S

PARAMS = {
    "grid": 1000,            # modulator pixels used (square)
    "pitch_um": 4.5,         # modulator pixel pitch (rig value)
    "wavelength_nm": 650.0,  # illumination (rig value, red)
    "z_cm": 10.0,            # Fresnel distance, near-eye order of magnitude
    "gs_iterations": 60,     # same budget both regimes, zero-phase init
    "patch_px": 100,         # pi-offset defect, centred
}
DEFAULT_TARGET = "cat2"      # headline candidate (silhouette unmistakable)
PHASE_CMAP = "gray"          # Dilawer 2026-08-05: grayscale, not twilight
PATCH_COLOUR = "#2ee06a"     # same green as the E8 ROI convention


def _load_target(name: str, n: int) -> np.ndarray:
    env = os.environ.get("SAILREV_STOCK")
    stock = Path(env) if env else \
        Path(os.environ["SAILREV_RESULTS"]) / "Stock Images" / "1000x1000"
    for ext in (".png", ".jpg", ".jpeg"):
        p = stock / f"{name}{ext}"
        if p.exists():
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
            if g.shape != (n, n):
                g = np.asarray(Image.fromarray(g.astype(np.uint8))
                               .resize((n, n)), dtype=np.float64)
            return g / (g.max() + 1e-12)
    raise FileNotFoundError(f"stock target {name} not found under {stock}")


def _targets() -> list[str]:
    """The manuscript's 18 targets, from the scored bench domain."""
    try:
        return sorted({r["target"] for r in S.records("bench", "ideal")})
    except Exception:
        return [DEFAULT_TARGET]


def _fields(n, pitch, lam, z):
    """(asm forward/backward, fourier forward, fourier backward)."""
    fx = np.fft.fftfreq(n, d=pitch)
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    arg = 1 - (lam * FX) ** 2 - (lam * FY) ** 2
    Hz = np.exp(1j * 2 * np.pi / lam * z * np.sqrt(np.maximum(arg, 0)))
    Hz *= (arg > 0)

    def asm(u, back=False):
        h = np.conj(Hz) if back else Hz
        return np.fft.ifft2(np.fft.fft2(u) * h)

    def ff(u):
        return np.fft.fftshift(np.fft.fft2(u)) / n

    def ff_inv(U):
        return np.fft.ifft2(np.fft.ifftshift(U)) * n

    return asm, ff, ff_inv


def _gs(amp_t, fwd, bwd, iters):
    phi = np.zeros_like(amp_t)   # zero-phase init: the smooth start
    for _ in range(iters):       # near-field practice actually uses
        u = fwd(np.exp(1j * phi))
        u = amp_t * np.exp(1j * np.angle(u))
        phi = np.angle(bwd(u))
    return phi


def _box_fraction(d, y0, x0, s, k=3):
    """Fraction of |dI| energy within a k-patch-width box about the patch."""
    n = d.shape[0]
    half = s * k // 2
    cy, cx = y0 + s // 2, x0 + s // 2
    box = d[max(0, cy - half):min(n, cy + half),
            max(0, cx - half):min(n, cx + half)]
    uniform = (min(s * k, n) ** 2) / (n * n)
    return box.sum() / d.sum(), uniform


def _one(target, out_dir, propagators, phase_cmap):
    """Compute and save the figure for one target; return caption metrics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = PARAMS["grid"]
    s = PARAMS["patch_px"]
    y0 = x0 = (n - s) // 2
    asm, ff, ff_inv = propagators

    tgt = _load_target(target, n)
    amp_t = np.sqrt(tgt)
    holo_nf = _gs(amp_t, lambda u: asm(u), lambda u: asm(u, True),
                  PARAMS["gs_iterations"])
    holo_ff = _gs(amp_t, ff, ff_inv, PARAMS["gs_iterations"])

    def recon_nf(p):
        return np.abs(asm(np.exp(1j * p))) ** 2

    def recon_ff(p):
        return np.abs(ff(np.exp(1j * p))) ** 2

    def psnr(r):
        r = r / r.max()
        return 10 * np.log10(1.0 / np.mean((r - tgt) ** 2))

    def pi_patch(p):
        q = p.copy()
        q[y0:y0 + s, x0:x0 + s] += np.pi
        return q

    d_nf = np.abs(recon_nf(pi_patch(holo_nf)) - recon_nf(holo_nf))
    d_ff = np.abs(recon_ff(pi_patch(holo_ff)) - recon_ff(holo_ff))
    f_nf, uniform = _box_fraction(d_nf, y0, x0, s)
    f_ff, _ = _box_fraction(d_ff, y0, x0, s)
    q_nf, q_ff = psnr(recon_nf(holo_nf)), psnr(recon_ff(holo_ff))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8))
    (a0, a1, a2), (b0, b1, b2) = axes

    a0.imshow(tgt, cmap="gray", vmin=0, vmax=1)
    a0.set_title("Target", fontsize=S.SIZE["title"], pad=6)
    im_ph = a1.imshow(holo_nf, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    a1.set_title(f"Near-field hologram, phase\n(Fresnel, z = "
                 f"{PARAMS['z_cm']:.0f} cm)",
                 fontsize=S.SIZE["title"], pad=6)
    a2.imshow(holo_ff, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    a2.set_title("Far-field hologram, phase\n(Fraunhofer)",
                 fontsize=S.SIZE["title"], pad=6)
    cb = fig.colorbar(im_ph, ax=[a1, a2], fraction=0.03, pad=0.05,
                      ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"],
                          fontsize=S.SIZE["tick_small"])

    # modulator plane: where the patch factually lives (marker kept here,
    # and ONLY here -- no box and no text on the |dI| panels, Dilawer
    # 2026-08-05, so nothing directs the eye)
    b0.imshow(np.full_like(tgt, 0.88), cmap="gray", vmin=0, vmax=1)
    b0.add_patch(Rectangle((x0, y0), s, s, fill=True,
                           facecolor=PATCH_COLOUR, alpha=0.6,
                           edgecolor=PATCH_COLOUR, linewidth=1.6))
    b0.text(0.5, 0.30, f"{s}$\\times${s} px patch\n"
            f"({s*PARAMS['pitch_um']/1e3:.1f} mm), $+\\pi$",
            ha="center", va="center", transform=b0.transAxes,
            fontsize=S.SIZE["annot"], color="#212121")
    b0.set_title("Modulator plane\n(perturbed patch, constant $\\pi$)",
                 fontsize=S.SIZE["title"], pad=6)
    vmax = float(max(np.percentile(d_nf, 99.9), np.percentile(d_ff, 99.9)))
    b1.imshow(d_nf, cmap="inferno", vmin=0, vmax=vmax)
    b1.set_title("$|\\Delta I|$, near field",
                 fontsize=S.SIZE["title"], pad=6)
    b2.imshow(d_ff, cmap="inferno", vmin=0, vmax=vmax)
    b2.set_title("$|\\Delta I|$, far field",
                 fontsize=S.SIZE["title"], pad=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    S.add_panel_label(a0, "a")
    S.add_panel_label(b0, "b")
    fig.subplots_adjust(wspace=0.05, hspace=0.16, left=0.02, right=0.87,
                        top=0.93, bottom=0.02)
    S.save(fig, Path(out_dir) / f"regimes_{target}")
    plt.close(fig)
    return {"f_nf": f_nf, "f_ff": f_ff, "uniform": uniform,
            "q_nf": q_nf, "q_ff": q_ff}


def build(out_dir, targets=None, phase_cmap=None):
    """One regime-comparison figure per target under out_dir/regimes/.

    targets: list of stems, default = the scored bench domain's 18.
    phase_cmap: default PHASE_CMAP (grayscale, Dilawer 2026-08-05).
    """
    S.apply_style()
    out = Path(out_dir) / "regimes"
    out.mkdir(parents=True, exist_ok=True)
    ts = list(targets) if targets else _targets()
    props = _fields(PARAMS["grid"], PARAMS["pitch_um"] * 1e-6,
                    PARAMS["wavelength_nm"] * 1e-9, PARAMS["z_cm"] * 1e-2)
    cmap = phase_cmap or PHASE_CMAP
    print(f"REGIMES | {len(ts)} targets, ~25 s each; params: "
          f"{PARAMS['grid']}x{PARAMS['grid']} @ {PARAMS['pitch_um']:g} um, "
          f"{PARAMS['wavelength_nm']:.0f} nm, z = {PARAMS['z_cm']:g} cm, "
          f"{PARAMS['gs_iterations']} GS iterations, zero-phase init")
    rows = []
    for t in ts:
        try:
            m = _one(t, out, props, cmap)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        rows.append((t, m))
        print(f"  {t:16s} |dI| near patch: near {100*m['f_nf']:5.1f}%  "
              f"far {100*m['f_ff']:5.1f}%  (uniform {100*m['uniform']:.0f}%)"
              f"   GS quality: near {m['q_nf']:.1f} / far {m['q_ff']:.1f} dB")
    return None
