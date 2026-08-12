"""
gs.py - Gerchberg-Saxton (GS) phase retrieval, strict Fraunhofer regime.

PORTED TO TORCH so GS shares the EXACT SAME forward-model physics as GD and
the transformer (physics.py: _slm_to_replay_forward / _replay_to_slm_adjoint),
including the physically-faithful pad_factor / apply_sinc / fill_factor path.
Previously this module was pure numpy with its own inline FFTs, which could
never pick up the faithful-model corrections applied to physics.py; that
inconsistency is what this rewrite fixes.

Implements the classical alternating-projection algorithm (Gerchberg & Saxton,
1972; see also Fienup, "Phase Retrieval Algorithms: A Comparison", 1982).

--- The adjoint subtlety (read this before touching pad_factor/apply_sinc) ---
GS's step (a) propagates the CURRENT replay-plane field estimate backward to
the SLM plane. With the bare, unpadded, no-envelope forward model this
backward step is simply the exact inverse of the forward FFT (ifft2 undoes
fft2 exactly for a unitary/"ortho" transform). Once a pixel-aperture sinc
envelope is introduced, the forward operator is no longer invertible: the sinc
has zeros, so dividing them back out is singular. The standard, correct fix
(used throughout computational imaging whenever a forward operator includes a
known, fixed, non-invertible envelope such as a pixel/pupil aperture) is to use
the ADJOINT operator instead of a naive inverse: since the sinc envelope is
real, it is self-adjoint, so the backward step multiplies by the envelope
AGAIN rather than dividing by it. Padding's adjoint is crop (extraction is the
transpose of zero-padding). See physics._replay_to_slm_adjoint for the exact
implementation and physics.py's own tests: this adjoint is an EXACT inverse
whenever apply_sinc=False (any pad_factor), and a well-posed, well-correlated
approximate inverse when apply_sinc=True (verified: >0.9 complex correlation
with the true field at fill_factor=0.91, since the sinc is close to 1 over
most of the passband and only attenuates -- it does not project to zero).

--- Consistency with gd.py (this is the point of the rewrite) ---
Both the algorithm's forward propagation (step c) and the per-iteration LOGGED
error use physics.py's shared operators with IDENTICAL pad_factor / apply_sinc
/ fill_factor / fill_is_areal to GD, so GS and GD convergence curves are
computed on the same forward model. The logged error is deliberately the SAME
formula GD uses for its own loss (MSE, sum-normalized to the padded pixel
count M*N, same target_formulation domain-squaring, same optional DC mask) --
see "target_formulation and the logged error" below.

Camera feedback and the logged error (GS+CITL): when magnitude_source is
provided, the logged error (and best-phase tracking) is computed from the
REAL camera capture vs target, not from the simulated forward-model
reconstruction -- mirroring gd.py's camera_source behavior exactly. Unlike
gd.py, GS needs no straight-through/detach construction to get there: GS
never calls .backward() (alternating projection, not gradient descent), so
there is no gradient graph to preserve -- the real measurement is used
directly. Without magnitude_source (plain simulation), the logged error uses
the simulated reconstruction, unchanged.

Design notes (carried over from the original numpy version):
  - No low-pass filtering, no off-axis carrier, by default (lowpass=False,
    add_carrier=False) -- these ARE implemented (see phase_lowpass_filter_torch
    and add_offaxis_carrier below) and available for other experiments, but the
    main GS-vs-GD-vs-transformer comparison reported in the manuscript runs
    with both disabled, matching Methods: "without low-pass filtering or
    carrier modulation".
  - Random phase initialization (init_phase="random"), matching Methods.
  - `binary_phase` exists for a separate binary-SLM comparison outside this paper
    and is NOT used in the main multilevel GS-vs-GD-vs-transformer comparison
    (binary_phase=False there).

target_amp and the target-formulation parameter (algorithm vs. logging):
  The `target_amp` parameter is named for what it has always contained in the
  published manuscript: an amplitude target (sqrt of intensity). This class
  also supports passing an intensity array directly into the same parameter,
  to compare amplitude- vs. intensity-domain optimization targets for GS.
  UNCHANGED FROM THE ORIGINAL: the core ALGORITHM does not know or care which
  domain `target_amp` is in -- it enforces `target_amp` (or, at padded
  resolution, its bicubic upsample) as the replay-plane magnitude constraint
  every iteration, verbatim, regardless of domain. This is a real consequence
  of the algorithm's structure, not a bug (see original module docstring
  discussion of why this makes the per-iteration constraint's numeric scale
  differ between conditions).
  NEW: `target_formulation` ("amplitude" default | "intensity") is used ONLY
  to correctly interpret target_amp when computing the LOGGED reconstruction
  error (mirroring gd.py's identical target_formulation handling exactly:
  target_intensity = target_amp**2 if "amplitude" else target_amp). This does
  NOT change the algorithm's magnitude-enforcement step in any way -- it only
  makes the logged/reported error comparable to GD's.

best-phase tracking (BEHAVIOUR CHANGE from the original numpy version):
  The original numpy GS had no per-iteration error signal at all and always
  returned whatever the FINAL iteration produced. This torch version computes
  a per-iteration error (see above) and tracks the BEST phase by that error,
  exactly mirroring gd.py's best_loss/best_phase/best_iteration convention
  (gd.py's own "phase_radians" key has always returned the BEST phase, not the
  final iterate). "phase_radians" in the returned dict is therefore now the
  BEST-tracked phase, matching gd.py, not necessarily the last iteration's
  phase. For plain simulation GS (no magnitude_source / camera feedback), GS
  converges close to monotonically, so best is typically indistinguishable
  from final. For GS+CITL (magnitude_source provided), this prevents the same
  class of "reported iteration is under-converged because raw camera noise
  made an early iteration's logged error dip" bug that was found and fixed for
  GD+CITL. This is a deliberate, flagged behaviour change -- any driver script
  reading gs_out["phase_radians"] now gets the best-tracked phase, not the
  final one.
"""
import time
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from physics import _slm_to_replay_forward, _replay_to_slm_adjoint
from stats_torch import phase_to_uint8


def phase_lowpass_filter_torch(phase: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """
    FFT-based low-pass filter for 2D phase arrays. Not used in the main reported
    GS comparison (see module docstring) -- available for other experiments that
    explicitly opt in via lowpass=True. Identical in form to gd.py's function of
    the same name (duplicated here rather than cross-imported, to keep gs.py
    self-contained; this is a generic auxiliary filter, unrelated to the shared
    forward-model physics that gd.py and gs.py otherwise share via physics.py).

    phase: (H, W) tensor in radians
    sigma: fraction of image size for circular low-pass radius
    """
    H, W = phase.shape
    cy, cx = H // 2, W // 2
    radius = max(1, int(min(H, W) * sigma))

    complex_phase = torch.exp(1j * phase)
    F = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(complex_phase), norm="ortho"))

    yy, xx = torch.meshgrid(
        torch.arange(H, device=phase.device),
        torch.arange(W, device=phase.device),
        indexing="ij",
    )
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    F_filtered = torch.zeros_like(F)
    F_filtered[mask] = F[mask]

    filtered_complex = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(F_filtered), norm="ortho")
    )
    return torch.angle(filtered_complex)


class GerchbergSaxton:
    def add_offaxis_carrier(self, phase: np.ndarray, fx: float = 0.1, fy: float = 0.1) -> np.ndarray:
        """
        Add a linear off-axis phase carrier. Not used in the main reported GS
        comparison (see module docstring) -- available for off-axis hologram
        experiments that explicitly opt in via add_carrier=True. Applied
        post-hoc to the final numpy phase array, exactly mirroring gd.py's
        identically-named method and its position in the return pipeline
        (after best-phase selection and lowpass, before uint8 conversion).
        """
        Ny, Nx = phase.shape
        y = np.arange(Ny) - Ny // 2
        x = np.arange(Nx) - Nx // 2
        X, Y = np.meshgrid(x, y)
        carrier = 2 * np.pi * (fx * X / Nx + fy * Y / Ny)
        return np.mod(phase + carrier, 2 * np.pi)

    def quantize_binary_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Quantize phase to binary values {0, pi}. Used only for the binary-SLM
        comparison outside this paper, not the main multilevel GS-vs-GD-vs-transformer
        comparison. Torch version (operates in-loop on SLM-plane phase tensors).
        """
        phase_wrapped = torch.angle(torch.exp(1j * phase))
        return torch.where(torch.cos(phase_wrapped) >= 0,
                           torch.zeros_like(phase_wrapped),
                           torch.full_like(phase_wrapped, float(np.pi)))

    def run(
        self,
        target_amp,
        iterations: int = 50,
        init_phase: str = "random",
        lowpass: bool = False,
        binary_phase: bool = False,
        lpf_sigma: float = 0.08,
        add_carrier: bool = False,
        carrierX: float = 0.1,
        carrierY: float = 0.1,
        show: bool = False,
        magnitude_source=None,
        device: str = "cuda",
        pad_factor: int = 1,
        apply_sinc: bool = False,
        fill_factor: float = 1.0,
        fill_is_areal: bool = True,
        target_formulation: str = "amplitude",
        dc_mask=None,
        log_csv_path: str | None = None,
        iteration_callback=None,
        seed: int | None = None,
    ):
        """
        Run phase-only Gerchberg-Saxton on a preprocessed square target, torch
        implementation sharing physics.py's forward/adjoint operators with GD.

        Algorithm (standard GS / alternating projections, see module docstring
        for references and the adjoint subtlety): starting from a random phase
        guess, repeatedly (a) propagate to the SLM/hologram plane via the
        ADJOINT of the shared forward operator, (b) enforce the phase-only
        constraint there, (c) propagate back to the replay plane via the shared
        FORWARD operator (same one GD uses, with the same pad_factor/
        apply_sinc/fill_factor), (d) enforce the target magnitude there while
        keeping the phase this iteration's forward propagation produced.

        Args:
            target_amp: (H, W) square target array, already preprocessed.
                NOTE: despite the name, this may be either an amplitude target
                (sqrt of intensity) or an intensity target (target_formulation
                controls how this is interpreted for LOGGING only -- see
                module docstring "target_amp and the target-formulation
                parameter"). The ALGORITHM applies whatever array it is given
                as the replay-plane magnitude constraint verbatim, unchanged
                from the original.
            iterations: number of GS iterations
            init_phase: "random" or "zeros"
            magnitude_source: optional callable,
                magnitude_source(phase_h_np: np.ndarray, iteration_index: int) -> np.ndarray,
                called every iteration AFTER the SLM-plane phase-only
                constraint (so it receives the current phase-only SLM pattern
                as a NUMPY array, for backward compatibility with existing
                camera-feedback driver scripts), and its numpy return value is
                used as the replay-plane magnitude constraint for that
                iteration -- the camera-feedback / GS+CITL hook. IMPORTANT: the
                returned array's shape must be (M,N) = (pad_factor*H,
                pad_factor*W), i.e. the SAME padded resolution the target and
                simulated reconstruction now live at when pad_factor>1 -- NOT
                (H,W). This mirrors gd.py's camera_source resolution
                requirement exactly (see gd.py's camera capture pipeline notes
                on upsampling out_hw to match). Default None: target_amp
                (upsampled to (M,N) if pad_factor>1) is used every iteration.
            device: "cuda" or "cpu".
            pad_factor / apply_sinc / fill_factor / fill_is_areal: forwarded,
                UNCHANGED IN MEANING, to physics.py's shared forward/adjoint
                operators -- identical to gd.py's parameters of the same name.
                Defaults (1, False, 1.0, True) reproduce behaviour
                mathematically equivalent to the original numpy GS (see
                physics.py: at pad_factor=1, apply_sinc=False, the shared
                forward/adjoint pair is an exact FFT/IFFT pair, matching
                classical GS exactly).
            target_formulation: "amplitude" (default) | "intensity". Used ONLY
                for the logged reconstruction error (mirrors gd.py exactly:
                target_intensity = target_amp**2 if "amplitude" else
                target_amp). Does NOT affect the algorithm's own magnitude
                enforcement, which is unchanged from the original and domain-
                agnostic by design.
            dc_mask: optional (M,N) or (1,M,N) boolean tensor, True inside the
                zero-order (DC) region to EXCLUDE from the LOGGED error only
                (mirrors gd.py's dc_mask exactly -- same shape requirement:
                built at the PADDED resolution (M,N) when pad_factor>1, with
                DC_CENTER/DC_RADIUS both scaled by pad_factor). Does not affect
                the algorithm itself. Raises a clear error on shape mismatch,
                same as gd.py.
            log_csv_path: optional path; if given, per-iteration error is
                written to this CSV (see module docstring "target_formulation
                and the logged error" -- columns mirror gd.py's convention as
                closely as GS's structure allows).
            iteration_callback: optional callable(record_dict), called every
                iteration with the same record written to the CSV.
            seed: optional int, seeds torch/numpy RNG for reproducibility.

        Returns:
            dict with:
                phase_radians: (H, W) float32 -- the BEST-tracked SLM-plane
                    phase (see module docstring "best-phase tracking" -- this
                    is a deliberate behaviour change from the original, which
                    always returned the final iterate; mirrors gd.py's
                    existing best-tracking convention).
                phase_uint8: (H, W) uint8
                best_iteration: int or None (None only if iterations=0)
                best_error: float or None
                loss_history: list of per-iteration logged errors
                iteration_records: list of per-iteration record dicts
        """
        if target_formulation not in ("amplitude", "intensity"):
            raise ValueError(
                f"Unknown target_formulation: {target_formulation!r}. "
                f"Expected 'amplitude' (published/default) or 'intensity'."
            )

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        A_np = np.asarray(target_amp, dtype=np.float32)
        if A_np.ndim != 2:
            raise ValueError(f"Expected target_amp shape (H,W), got {A_np.shape}")
        H, W = A_np.shape
        if H != W:
            raise ValueError(f"GS expects square target_amp, got {A_np.shape}")

        if not (isinstance(pad_factor, int) and pad_factor >= 1):
            raise ValueError(f"pad_factor must be an integer >= 1, got {pad_factor!r}")

        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        A = torch.from_numpy(A_np).to(dev)  # (H,W) real -- the enforced magnitude, domain-agnostic

        if show:
            plt.figure(figsize=(4, 4))
            plt.imshow(A_np, cmap="gray")
            plt.title("Target amplitude" if target_formulation == "amplitude" else "Target (as passed)")
            plt.axis("off")
            plt.show()

        # ---- Padded-resolution setup (mirrors gd.py's target-upsampling fix) ----
        # A_padded is the array actually enforced as the replay-plane magnitude
        # each iteration (unless magnitude_source overrides it) -- upsampled to
        # (M,N) if pad_factor>1, since the replay plane now lives at that
        # resolution (physics.py returns the replay uncropped -- see its
        # docstring). This is domain-agnostic, matching the algorithm's
        # original behaviour: whatever target_amp represents, its upsample is
        # what gets enforced, unchanged in kind.
        if pad_factor > 1:
            M, N = H * pad_factor, W * pad_factor
            A_padded = torch.nn.functional.interpolate(
                A.unsqueeze(0).unsqueeze(0), size=(M, N), mode="bicubic", align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            M, N = H, W
            A_padded = A

        # ---- Target intensity for the LOGGED error only (mirrors gd.py exactly) ----
        if target_formulation == "amplitude":
            target_intensity_for_log = A_padded ** 2
        else:
            target_intensity_for_log = A_padded
        target_intensity_for_log = target_intensity_for_log.unsqueeze(0)  # (1,M,N)
        target_for_loss = (
            target_intensity_for_log
            / (target_intensity_for_log.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            * (M * N)
        )

        # ---- Initial phase guess, generated directly at (M,N) (see module
        # docstring: the initial replay-plane field lives at whatever
        # resolution the replay plane is, from the very first iteration) ----
        if isinstance(init_phase, np.ndarray):
            if init_phase.shape != (H, W):
                raise ValueError(f"init_phase array shape {init_phase.shape} != expected SLM-plane ({H},{W})")
            seed_phase_t = torch.from_numpy(init_phase).to(device=dev, dtype=torch.float32).unsqueeze(0)
            h_seed = torch.polar(torch.ones_like(seed_phase_t), seed_phase_t)  # (1,H,W) complex, unit magnitude
            F_seed, _ = _slm_to_replay_forward(h_seed, pad_factor, apply_sinc, fill_factor, fill_is_areal)
            phase_guess = torch.angle(F_seed)[0]  # (M,N) - correct replay-plane starting phase
        elif init_phase == "random":
            phase_guess = 2 * np.pi * torch.rand(M, N, device=dev, dtype=torch.float32)
        elif init_phase == "zeros":
            phase_guess = torch.zeros(M, N, device=dev, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown init_phase: {init_phase}")

        if binary_phase:
            phase_guess = self.quantize_binary_phase(phase_guess)

        # Initial replay-plane field: enforced magnitude with the random/zero
        # phase guess, at (M,N), batched to (1,M,N) for the adjoint operator.
        F = torch.polar(A_padded, phase_guess).unsqueeze(0)

        # ---- Logging setup (mirrors gd.py's CSV/history pattern) ----
        iter_records = []
        loss_history = []
        best_error = float("inf")
        best_iteration = None
        best_phase_h = None
        t_run0 = time.perf_counter()

        csv_file = None
        csv_writer = None
        if log_csv_path is not None:
            csv_file = open(log_csv_path, "w", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=["iteration", "error", "best_error", "is_best", "elapsed_s"],
            )
            csv_writer.writeheader()

        h = None  # SLM-plane phase-only field, set inside the loop
        phase_h = None

        for iteration_index in tqdm(range(iterations), desc="Gerchberg-Saxton"):
            # Step (a): ADJOINT propagate replay-plane field estimate -> SLM plane.
            # See module docstring "adjoint subtlety": this is an EXACT inverse
            # when apply_sinc=False, and the correct (self-adjoint-envelope)
            # well-posed backward step when apply_sinc=True.
            h_est = _replay_to_slm_adjoint(F, pad_factor, apply_sinc, fill_factor, fill_is_areal, H, W)
            phase_h = torch.angle(h_est)  # (1,H,W)

            if binary_phase:
                phase_h = self.quantize_binary_phase(phase_h)

            # Step (b): phase-only constraint at the SLM plane.
            h = torch.polar(torch.ones_like(phase_h), phase_h)  # (1,H,W) complex, unit magnitude

            # Step (c): FORWARD propagate the (now phase-only) SLM field to the
            # replay plane, using the SAME shared operator (and pad/sinc/fill
            # settings) as GD. Always needed regardless of camera feedback --
            # its PHASE feeds step (d) below either way; only its MAGNITUDE is
            # ever replaced (by A_padded in plain simulation, or by the real
            # camera capture in GS+CITL).
            F_full, (M_chk, N_chk) = _slm_to_replay_forward(h, pad_factor, apply_sinc, fill_factor, fill_is_areal)

            recon_I = F_full.abs() ** 2
            recon_for_loss = recon_I / (recon_I.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (M * N)

            # Camera-feedback hook: if magnitude_source is provided, capture
            # NOW (moved earlier than the original step (d) position) so the
            # SAME real measurement can be used for BOTH the logged error
            # below AND the magnitude enforcement in step (d) -- one capture
            # per iteration either way, just reordered.
            #
            # GS never calls .backward() (alternating projection, not
            # gradient descent), so unlike gd.py's straight-through estimator
            # (needed there only to give backprop a differentiable path),
            # GS needs no such construction: when a real camera capture is
            # available, the logged error IS the real capture vs target,
            # directly. This mirrors gd.py's trust model for camera_source --
            # shape-checked here, not re-normalized -- the caller is expected
            # to already provide correctly (M*N)-sum-normalized data, exactly
            # as gd.py expects of camera_source.
            if magnitude_source is not None:
                phase_h_np = phase_h.squeeze(0).detach().cpu().numpy()
                magnitude_np = np.asarray(
                    magnitude_source(phase_h_np, iteration_index), dtype=np.float32
                )
                if magnitude_np.shape != (M, N):
                    raise ValueError(
                        f"magnitude_source returned shape {magnitude_np.shape}, "
                        f"expected ({M},{N}) (pad_factor={pad_factor}). When "
                        f"pad_factor>1 the callback must return its measurement "
                        f"resized to the PADDED resolution, matching how the "
                        f"camera capture pipeline must be updated (out_hw=(M,N))."
                    )
                magnitude_this_iter = torch.from_numpy(magnitude_np).to(dev).unsqueeze(0)

                # magnitude_source returns amplitude or intensity depending on
                # target_formulation (mirrors how target_amp itself was
                # supplied) -- square to intensity here if needed, exactly
                # matching target_for_loss's own domain handling above.
                if target_formulation == "amplitude":
                    value_for_loss = magnitude_this_iter ** 2
                else:
                    value_for_loss = magnitude_this_iter
            else:
                magnitude_this_iter = A_padded.unsqueeze(0)
                value_for_loss = recon_for_loss

            # --- Logged error: the REAL camera capture vs target when GS+CITL
            # (magnitude_source given), or the simulated reconstruction vs
            # target for plain simulation GS -- exactly mirroring gd.py's own
            # camera_source-vs-simulation choice for its logged loss. Same
            # formula either way: MSE (optionally DC-masked) between
            # sum-normalized-to-(M*N) value_for_loss and target_for_loss. ---
            if dc_mask is not None:
                mask_hw = dc_mask.shape[-2:]
                if tuple(mask_hw) != (M, N):
                    raise ValueError(
                        f"dc_mask has shape {tuple(dc_mask.shape)} (last two dims "
                        f"{tuple(mask_hw)}) but the replay plane is at resolution "
                        f"({M},{N}) (pad_factor={pad_factor}). Rebuild dc_mask at "
                        f"the SAME padded resolution, with DC_CENTER and DC_RADIUS "
                        f"both scaled by pad_factor."
                    )
                keep = (~dc_mask).to(dtype=value_for_loss.dtype, device=value_for_loss.device)
                while keep.ndim < value_for_loss.ndim:
                    keep = keep.unsqueeze(0)
                error_t = (keep * (value_for_loss - target_for_loss) ** 2).sum() / (keep.sum() + 1e-8)
            else:
                error_t = torch.mean((value_for_loss - target_for_loss) ** 2)

            current_error = float(error_t.item())
            is_best = current_error < best_error
            if is_best:
                best_error = current_error
                best_iteration = iteration_index
                best_phase_h = phase_h.detach().clone()

            loss_history.append(current_error)
            record = {
                "iteration": iteration_index,
                "error": current_error,
                "best_error": float(best_error),
                "is_best": bool(is_best),
                "elapsed_s": time.perf_counter() - t_run0,
            }
            iter_records.append(record)
            if csv_writer is not None:
                csv_writer.writerow(record)
                csv_file.flush()
            if iteration_callback is not None:
                iteration_callback(record)

            # Step (d): enforce the target magnitude in the replay plane,
            # keeping the phase this iteration's forward propagation produced.
            # magnitude_this_iter was already resolved above (real camera
            # capture, if provided, or A_padded otherwise) -- no second call
            # to magnitude_source here.
            F = torch.polar(magnitude_this_iter, torch.angle(F_full))

        if best_phase_h is None:
            # iterations=0 (or pathological): fall back to whatever phase_h was
            # last computed (mirrors gd.py's equivalent fallback).
            best_phase_h = phase_h if phase_h is not None else torch.zeros(1, H, W, device=dev)
            best_iteration = None

        best_phase = best_phase_h.squeeze(0)  # (H,W)

        if binary_phase:
            best_phase = self.quantize_binary_phase(best_phase)

        if lowpass:
            best_phase = phase_lowpass_filter_torch(best_phase, sigma=lpf_sigma)
            if binary_phase:
                best_phase = self.quantize_binary_phase(best_phase)

        phase_np = best_phase.detach().cpu().numpy().astype(np.float32)

        if add_carrier:
            phase_np = self.add_offaxis_carrier(phase_np, fx=carrierX, fy=carrierY)
            if binary_phase:
                phase_np = np.where(np.cos(phase_np) >= 0, 0.0, np.pi).astype(np.float32)

        phase_np = np.angle(np.exp(1j * phase_np)).astype(np.float32)

        if binary_phase:
            phase_np = np.where(np.cos(phase_np) >= 0, 0.0, np.pi).astype(np.float32)

        phase_8bit = phase_to_uint8(phase_np)

        if csv_file is not None:
            csv_file.close()

        return {
            "phase_radians": phase_np,
            "phase_uint8": phase_8bit,
            "best_iteration": best_iteration,
            "best_error": (None if best_iteration is None else float(best_error)),
            "loss_history": loss_history,
            "iteration_records": iter_records,
        }