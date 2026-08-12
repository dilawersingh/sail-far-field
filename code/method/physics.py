import torch
import math

def unit_magnitude_field(Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Y: (B, 2, H, W) where channels are a_raw, b_raw
    returns: U complex tensor (B, H, W) with |U| ~ 1
    """
    if Y.ndim != 4 or Y.shape[1] != 2:
        raise ValueError(f"Expected Y shape (B,2,H,W), got {Y.shape}")

    a_raw = Y[:, 0]
    b_raw = Y[:, 1]
    q = torch.sqrt(a_raw * a_raw + b_raw * b_raw) + eps
    a = a_raw / q
    b = b_raw / q
    U = torch.complex(a, b)
    return U

def _make_sinc_envelope(M: int, N: int, device, dtype,
                        fill_factor: float = 1.0, fill_is_areal: bool = True):
    """
    Pixel-aperture sinc envelope on the padded replay grid.

    Base convention (fill_factor=1):
        v = (arange(M) - M//2) / M          # spans ~[-0.5, 0.5)
        u = (arange(N) - N//2) / N
        envelope = sinc(v) * sinc(u)
    -> first zeros at the padded-field edges (+/-1 diffraction orders),
       i.e. active aperture equals the full pixel pitch.

    Fill factor: real SLM pixels have an active (light-modulating) region
    smaller than the pixel pitch. The sinc argument scales with the ACTIVE
    APERTURE WIDTH, so:

        envelope = sinc(ff_lin * v) * sinc(ff_lin * u)

    where ff_lin is the LINEAR (per-axis) fill fraction = active_width / pitch.
    Lower fill -> smaller argument -> WIDER sinc -> LESS in-field attenuation
    (first zeros move beyond the field edge). fill_factor=1 recovers the base
    convention exactly.

    fill_is_areal:
        Datasheets usually quote AREAL fill factor (fraction of pixel AREA that
        is active). If fill_is_areal=True (default), fill_factor is treated as
        areal and converted to linear via ff_lin = sqrt(fill_factor) (assuming
        a square active region). If your value is already the per-axis linear
        fill, pass fill_is_areal=False to use it directly.

    This is an AMPLITUDE envelope (applied to the complex field before
    squaring). Returned fftshift-centred. Returns (M, N) real tensor.
    """
    if not (0.0 < fill_factor <= 1.0):
        raise ValueError(f"fill_factor must be in (0,1], got {fill_factor}")
    ff_lin = (fill_factor ** 0.5) if fill_is_areal else fill_factor

    v = (torch.arange(M, device=device, dtype=dtype) - (M // 2)) / M
    u = (torch.arange(N, device=device, dtype=dtype) - (N // 2)) / N
    V, Uc = torch.meshgrid(v, u, indexing="ij")
    return torch.sinc(ff_lin * V) * torch.sinc(ff_lin * Uc)


def _slm_to_replay_forward(U: torch.Tensor, pad_factor: int, apply_sinc: bool,
                           fill_factor: float, fill_is_areal: bool):
    """
    SHARED forward operator: SLM-plane complex field -> replay-plane complex
    field. This is the single source of truth for the "physically faithful"
    Fraunhofer forward model -- used by hologram_intensity_from_field/_phase
    AND by the torch Gerchberg-Saxton implementation (gs.py), so GD, GS, and
    the transformer all propagate through IDENTICAL physics.

    U: (B,H,W) complex, unit magnitude (or phase-only) SLM-plane field.

    Steps: zero-pad to (pad_factor*H, pad_factor*W) -> centred FFT
    (ifftshift -> fft2(ortho) -> fftshift) -> optional amplitude sinc envelope.

    Returns: (F_centered, (M,N)) where F_centered is (B,M,N) complex and
    (M,N) = (pad_factor*H, pad_factor*W).
    """
    B, H, W = U.shape
    if pad_factor > 1:
        M, N = H * pad_factor, W * pad_factor
        pad_y = (M - H) // 2
        pad_x = (N - W) // 2
        U_pad = torch.nn.functional.pad(U, (pad_x, pad_x, pad_y, pad_y),
                                        mode="constant", value=0.0)
        F_centered = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(U_pad, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
    else:
        M, N = H, W
        # ifftshift input here too (matching the pad>1 branch's convention) --
        # this does NOT change the resulting intensity |F|^2 (verified: an
        # input ifftshift only introduces a linear phase ramp on F, never
        # changes |F|), but it DOES make this general-purpose helper's forward
        # operator self-consistent with its own adjoint (_replay_to_slm_adjoint)
        # across all pad_factor values. hologram_intensity_from_field's
        # separately-preserved byte-identical fast path does NOT call this
        # helper for pad_factor=1/apply_sinc=False, so that guarantee is
        # unaffected by this choice.
        F_centered = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(U, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )

    if apply_sinc:
        env_amp = _make_sinc_envelope(M, N, device=U.device, dtype=F_centered.real.dtype,
                                      fill_factor=fill_factor, fill_is_areal=fill_is_areal)
        F_centered = F_centered * env_amp.unsqueeze(0)

    return F_centered, (M, N)


def _replay_to_slm_adjoint(F_centered: torch.Tensor, pad_factor: int, apply_sinc: bool,
                           fill_factor: float, fill_is_areal: bool, H: int, W: int):
    """
    ADJOINT of _slm_to_replay_forward: replay-plane complex field -> SLM-plane
    complex field ESTIMATE. This is NOT a naive inverse -- the forward operator
    includes a sinc envelope with zeros, so dividing it back out is singular
    and numerically unstable. The correct, standard approach for a known,
    fixed, non-invertible system operator (the sinc here plays the same role
    as a known aperture/PSF in computational imaging) is to apply its ADJOINT:
    for a real diagonal envelope S, S^H = S (self-adjoint, since it's real),
    so the backward step multiplies by S AGAIN rather than dividing by it.
    Padding's adjoint is crop (the transpose of zero-padding is extraction).
    FFT under norm="ortho" is unitary, so its adjoint is the inverse FFT.

    Forward:  U -> pad -> ifftshift -> fft2(ortho) -> fftshift -> * S = F
    Adjoint:  F -> * S -> ifftshift -> ifft2(ortho) -> fftshift -> crop = U_est

    This is what makes Gerchberg-Saxton's "propagate back to the SLM plane"
    step well-posed once pad_factor > 1 and/or apply_sinc=True are used inside
    the alternating-projection loop (see gs.py). Used with pad_factor=1 and
    apply_sinc=False, this collapses to the exact original ifft2(...,
    norm="ortho"), matching classical GS.

    F_centered: (B,M,N) complex, at the SAME (M,N) that
                _slm_to_replay_forward would have produced for this pad_factor.
    Returns: (B,H,W) complex SLM-plane field estimate.
    """
    M, N = F_centered.shape[-2], F_centered.shape[-1]

    if apply_sinc:
        env_amp = _make_sinc_envelope(M, N, device=F_centered.device, dtype=F_centered.real.dtype,
                                      fill_factor=fill_factor, fill_is_areal=fill_is_areal)
        F_for_inverse = F_centered * env_amp.unsqueeze(0)  # self-adjoint: multiply again
    else:
        F_for_inverse = F_centered

    U_pad = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(F_for_inverse, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )

    if pad_factor > 1:
        U_est = center_crop_2d(U_pad, H, W)  # adjoint of zero-pad is crop
    else:
        U_est = U_pad

    return U_est


def hologram_intensity_from_field(
    Y: torch.Tensor,
    eps: float = 1e-6,
    return_field: bool = False,
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    Far-field (Fraunhofer) forward model. Thin wrapper around the SHARED
    forward operator _slm_to_replay_forward (also used by torch GS, see gs.py),
    so every method in the codebase propagates through identical physics.

    Base path (pad_factor=1, apply_sinc=False) -- byte-identical to the
    original pre-refactor behaviour:

        Y -> unit magnitude complex field U
          -> F = fft2(U, norm="ortho")
          -> I = fftshift(|F|^2)

    Physically-faithful path (pad_factor=2, apply_sinc=True):

        U -> zero-pad SLM plane to (pad_factor*H, pad_factor*W)
          -> centred FFT (ifftshift -> fft2(norm="ortho") -> fftshift)
          -> multiply the COMPLEX field by the amplitude sinc envelope
          -> I = |F_enveloped|^2

    IMPORTANT -- resolution convention: when pad_factor > 1 the replay is
    returned at the PADDED resolution (pad_factor*H, pad_factor*W); it is NOT
    cropped back to (H,W). This preserves the finer replay sampling that the
    padding buys (the whole point: the intensity |F|^2 has ~2x the field
    bandwidth, so it needs the denser grid to avoid aliasing). Callers must
    therefore supply the target (and, for CITL, the camera image) at the SAME
    padded resolution -- e.g. upsample the target with
    F.interpolate(..., size=(pad_factor*H, pad_factor*W), mode="bicubic").
    Physical replay extent is unchanged by padding; only the sampling density
    increases (padding grows the SLM-plane extent, which shrinks the replay
    sample spacing, while the SLM pixel pitch -- hence replay extent -- is
    fixed).

    The sinc is applied as an AMPLITUDE envelope to the complex field BEFORE
    squaring, so field and intensity are consistent by construction
    (|F * sinc|^2 = |F|^2 * sinc^2 automatically). Envelope convention: first
    zeros at the padded-field edges, fill factor = 1 (see _make_sinc_envelope).

    Args:
        Y: (B,2,H,W) predicted field
        eps: magnitude stabilization
        return_field: if True, also return the (enveloped) complex replay field
        pad_factor: integer >= 1. 1 = original behaviour (no padding).
        apply_sinc: if True, apply the pixel amplitude sinc envelope.
        fill_factor / fill_is_areal: see _make_sinc_envelope.

    Returns:
        I_centered  (B, pad_factor*H, pad_factor*W) if return_field=False
        (I_centered, F_centered) if return_field=True
        (When pad_factor=1 the output is (B,H,W) as before.)
    """
    if not (isinstance(pad_factor, int) and pad_factor >= 1):
        raise ValueError(f"pad_factor must be an integer >= 1, got {pad_factor!r}")

    # ---- Fast path: exact original behaviour, byte-identical to pre-change ----
    if pad_factor == 1 and not apply_sinc:
        U = unit_magnitude_field(Y, eps=eps)
        F = torch.fft.fft2(U, norm="ortho")
        I_unshifted = (F.abs() ** 2)
        I_centered = torch.fft.fftshift(I_unshifted, dim=(-2, -1))
        if return_field:
            F_centered = torch.fft.fftshift(F, dim=(-2, -1))
            return I_centered, F_centered
        return I_centered

    # ---- Physically-faithful path (delegates to the shared forward operator) ----
    U = unit_magnitude_field(Y, eps=eps)  # (B,H,W) complex
    F_centered, _ = _slm_to_replay_forward(U, pad_factor, apply_sinc, fill_factor, fill_is_areal)
    I_centered = F_centered.abs() ** 2

    if return_field:
        return I_centered, F_centered
    return I_centered

def phase_to_field(phase: torch.Tensor) -> torch.Tensor:
    """
    phase: (B,H,W) or (H,W) in radians
    returns: (B,2,H,W) with channels [cos(phase), sin(phase)]
    """
    if phase.ndim == 2:
        phase = phase.unsqueeze(0)
    if phase.ndim != 3:
        raise ValueError(f"Expected phase shape (H,W) or (B,H,W), got {tuple(phase.shape)}")

    real = torch.cos(phase)
    imag = torch.sin(phase)
    return torch.stack([real, imag], dim=1)

def hologram_intensity_from_phase(
    phase: torch.Tensor,
    eps: float = 1e-6,
    return_field: bool = False,
    pad_factor: int = 1,
    apply_sinc: bool = False,
    fill_factor: float = 1.0,
    fill_is_areal: bool = True,
):
    """
    phase: (B,H,W) or (H,W) in radians
    returns replay intensity using the same physics path as hologram_intensity_from_field

    pad_factor / apply_sinc / fill_factor / fill_is_areal forwarded unchanged;
    defaults (1, False, 1.0, True) are byte-identical to the original. When
    pad_factor > 1 the replay is returned at padded resolution (see
    hologram_intensity_from_field docstring) -- the caller must match
    target/camera resolution accordingly.
    """
    Y = phase_to_field(phase)
    return hologram_intensity_from_field(
        Y, eps=eps, return_field=return_field,
        pad_factor=pad_factor, apply_sinc=apply_sinc,
        fill_factor=fill_factor, fill_is_areal=fill_is_areal,
    )

def zero_pad_field(Y: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
    """
    Y: (B, 2, H, W)
    returns centered zero-padded field: (B, 2, H_out, W_out)
    """
    B, C, H, W = Y.shape
    if C != 2:
        raise ValueError(f"Expected channel dim 2, got {Y.shape}")
    if H > H_out or W > W_out:
        raise ValueError(f"Input {(H,W)} cannot fit into output {(H_out,W_out)}")

    out = torch.zeros((B, C, H_out, W_out), dtype=Y.dtype, device=Y.device)
    y0 = (H_out - H) // 2
    x0 = (W_out - W) // 2
    out[:, :, y0:y0+H, x0:x0+W] = Y
    return out

def center_crop_2d(
    x: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """
    Center-crop the last two dims.

    x: (..., H, W)
    returns: (..., out_h, out_w)
    """
    H, W = x.shape[-2:]
    if out_h > H or out_w > W:
        raise ValueError(f"Crop size {(out_h, out_w)} must be <= input size {(H, W)}")

    y0 = (H - out_h) // 2
    x0 = (W - out_w) // 2
    return x[..., y0:y0+out_h, x0:x0+out_w]

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap phase differences to [-pi, pi).

    Args:
        x: tensor of phase differences

    Returns:
        wrapped tensor in [-pi, pi)
    """
    return (x + math.pi) % (2 * math.pi) - math.pi


def detect_vortices_from_phase(phase: torch.Tensor, threshold: float = math.pi):
    """
    Detect optical vortices from a 2D phase map using plaquette winding.

    A vortex is identified when the wrapped phase change around a 1x1 cell
    sums to approximately +2pi or -2pi.

    Args:
        phase: (H, W) tensor of phase values in radians
        threshold: minimum absolute winding magnitude to count as a vortex.
                   Default pi is conservative; exact vortices are near 2pi.

    Returns:
        pos_mask: (H-1, W-1) bool tensor, True where a + vortex is detected
        neg_mask: (H-1, W-1) bool tensor, True where a - vortex is detected
        winding:  (H-1, W-1) tensor of summed wrapped phase around each plaquette
    """
    if phase.ndim != 2:
        raise ValueError(f"Expected phase shape (H, W), got {tuple(phase.shape)}")

    # Corners of each plaquette:
    # p00 --- p01
    #  |       |
    # p10 --- p11
    p00 = phase[:-1, :-1]
    p01 = phase[:-1, 1:]
    p11 = phase[1:, 1:]
    p10 = phase[1:, :-1]

    # Wrapped phase differences around the plaquette
    d1 = wrap_to_pi(p01 - p00)  # top edge
    d2 = wrap_to_pi(p11 - p01)  # right edge
    d3 = wrap_to_pi(p10 - p11)  # bottom edge
    d4 = wrap_to_pi(p00 - p10)  # left edge

    winding = d1 + d2 + d3 + d4

    pos_mask = winding > threshold
    neg_mask = winding < -threshold

    return pos_mask, neg_mask, winding


def detect_vortices_from_field(field: torch.Tensor, threshold: float = math.pi):
    """
    Detect optical vortices directly from a complex field.

    Args:
        field: either
            - complex tensor of shape (H, W), or
            - real tensor of shape (2, H, W) where [0]=real, [1]=imag
        threshold: winding threshold

    Returns:
        pos_mask: (H-1, W-1) bool tensor
        neg_mask: (H-1, W-1) bool tensor
        winding:  (H-1, W-1) tensor
        phase:    (H, W) phase map used for detection
    """
    if torch.is_complex(field):
        phase = torch.angle(field)
    else:
        if field.ndim != 3 or field.shape[0] != 2:
            raise ValueError(
                f"Expected complex field (H,W) or real field (2,H,W), got {tuple(field.shape)}"
            )
        real = field[0]
        imag = field[1]
        phase = torch.atan2(imag, real)

    pos_mask, neg_mask, winding = detect_vortices_from_phase(phase, threshold=threshold)
    return pos_mask, neg_mask, winding, phase