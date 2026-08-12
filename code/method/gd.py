"""
gd.py - Gradient-descent (GD) phase retrieval, strict Fraunhofer regime.

Implements direct gradient-based phase-only hologram optimization: the SLM phase
is a free parameter, optimized by backpropagating an intensity-domain MSE loss
through the differentiable Fraunhofer forward model in physics.py
(hologram_intensity_from_phase). This mirrors the GD baseline described in
manuscript Methods ("GD, optimization was performed with the Adam optimizer...
intensity-based loss, with no total variation regularization, low-pass filtering,
or carrier term").

Design notes:
  - Adam optimizer, lr=0.03, no TV regularization (tv_weight=0.0), no low-pass
    filter, no off-axis carrier, by default -- matches the published comparison
    exactly. TV/lowpass/carrier ARE implemented for other experiments but are not
    enabled in the main GS-vs-GD-vs-transformer comparison.
  - Random phase initialization (init_phase="random"), matching Methods.
  - The optimizer tracks and returns the *best* phase by training loss across all
    iterations (best_phase/best_loss), not simply the final iterate -- this is an
    asymmetry relative to gs.py's GerchbergSaxton.run(), which has no equivalent
    best-iterate tracking and always returns whatever phase exists when the loop
    ends. Worth keeping in mind if GS's iteration-to-iteration loss is ever
    non-monotonic (e.g. under the intensity-target condition, where the
    replay-plane constraint is scaled differently -- see target_formulation below):
    GD's reported numbers benefit from best-iterate selection in a way GS's do not.
    The returned dict also includes "best_iteration" (the 0-indexed loop
    iteration k at which best_loss/best_phase were last updated, matching the
    same k passed to camera_source -- so a caller using camera_source to save
    per-iteration captures can look up exactly which physical measurement
    corresponds to the returned phase, rather than only having the phase array
    with no link back to the iteration that produced it). None if the loop
    never updated best_loss from its initial value.

target_amp, the internal A**2 step, and the target_formulation parameter:
  Unlike gs.py's GerchbergSaxton (which treats its target parameter as a raw
  replay-plane magnitude constraint, agnostic to whether it represents amplitude
  or intensity), this class's run() has historically ALWAYS internally computed
  `target_intensity_np = A ** 2`, i.e. it assumes whatever is passed in is an
  AMPLITUDE and squares it to recover intensity for its loss. This is correct
  when `target_amp` is genuinely `sqrt(intensity)` (the published convention).

  That assumption becomes WRONG -- not just a different valid choice -- if a
  true intensity array is passed in and then squared again: the loss target
  would become intensity**2, not intensity, a unit error rather than a
  meaningful comparison. The `target_formulation` parameter below makes the
  squaring conditional on what the caller says `target_amp` actually contains,
  so both domains can be passed in correctly. GS needs no equivalent parameter,
  since it never assumes a domain for its target in the first place -- this
  asymmetry between the two classical baselines' original implementations is
  the reason both files were traced carefully when this parameter was added,
  rather than only changing where the target array is constructed.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import csv
import time

from physics import hologram_intensity_from_phase
from stats_torch import phase_to_uint8


def phase_lowpass_filter_torch(phase: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """
    FFT-based low-pass filter for 2D phase arrays. Not used in the main reported
    GD comparison (see module docstring) -- available for other experiments that
    explicitly opt in via lowpass=True.
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


class GradientDescentHologram:
    def add_offaxis_carrier(self, phase: np.ndarray, fx: float = 0.1, fy: float = 0.1) -> np.ndarray:
        """Not used in the main reported GD comparison (see module docstring)."""
        H, W = phase.shape
        y = np.arange(H) - H // 2
        x = np.arange(W) - W // 2
        X, Y = np.meshgrid(x, y)
        carrier = 2 * np.pi * (fx * X / W + fy * Y / H)
        return phase + carrier

    @staticmethod
    def normalize_01_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Min-max normalize to [0,1]. Currently unused by run() -- retained because
        it implements the "mse01" loss_type that run()'s docstring has historically
        described, even though that branch was not actually wired up (see note at
        the loss_type parameter below). Kept rather than deleted in case "mse01"
        is wired up deliberately in a future revision.
        """
        x = x - x.amin(dim=(-2, -1), keepdim=True)
        x = x / (x.amax(dim=(-2, -1), keepdim=True) + eps)
        return x

    def run(
        self,
        target_amp,
        iterations: int = 2000,
        init_phase: str = "random",
        lr: float = 0.03,
        optimizer_name: str = "adam",
        loss_type: str = "mse_sum",
        target_formulation: str = "amplitude",
        tv_weight: float = 0.0,
        lowpass: bool = False,
        binary_phase: bool=False,
        lpf_sigma: float = 0.08,
        add_carrier: bool = False,
        carrierX: float = 0.1,
        carrierY: float = 0.1,
        seed: int | None = None,
        device: str = "cuda",
        show: bool = False,
        camera_source=None,
        dc_mask=None,
        log_csv_path: str | None = None,
        iteration_callback=None,
        pad_factor: int = 1,
        apply_sinc: bool = False,
        fill_factor: float = 1.0,
        fill_is_areal: bool = True,
    ):
        """
        Gradient descent for phase-only hologram optimization.

        Args:
            target_amp: (H, W) target array. What it represents is controlled by
                target_formulation (below) -- despite the parameter's name, this
                is NOT always an amplitude array; see target_formulation.
            iterations: optimization steps
            init_phase: "random" or "zeros"
            lr: learning rate
            optimizer_name: "adam" or "sgd"
            loss_type: retained for interface compatibility / documentation of
                intent ("mse01": MSE after 0-1 normalization: "mse_sum": MSE
                after sum normalization), but NOTE -- as of this revision the
                live code path always performs sum-normalization regardless of
                this value (this was already true before this revision; it has
                been made explicit here rather than left as dead, misleading
                commented-out branches -- see git history / README for the
                published GD results, which were all generated with the
                sum-normalization path, matching the default below). Changed
                default from "mse01" to "mse_sum" to match what the code has
                always actually done, rather than what it claimed to accept.
            target_formulation: "amplitude" (default) | "intensity".
                Controls whether target_amp is squared to obtain the intensity
                training target ("amplitude": target_intensity = target_amp**2,
                the published/original behaviour, correct when target_amp is
                truly sqrt(intensity)) or used directly as the intensity training
                target with no squaring ("intensity": target_intensity =
                target_amp, for the intensity-target condition, correct when
                target_amp is already an intensity array). Passing an intensity
                array with target_formulation left at "amplitude" would silently
                square it a second time -- this parameter exists specifically to
                prevent that; see module docstring.
            tv_weight: optional TV regularization on phase
            camera_source: optional callable, signature
                camera_source(phase_for_forward: torch.Tensor, iteration_index: int)
                -> torch.Tensor (same shape as recon_for_loss, i.e. (1,H,W),
                normalized the same way -- sum-normalized * H*W).
                If provided, called every iteration with the current phase
                (exactly what would be displayed on the SLM this iteration),
                and used to build a straight-through proxy for the loss:
                    I_proxy = recon_for_loss + (I_camera - recon_for_loss).detach()
                i.e. the loss VALUE comes from the camera measurement, while
                gradients in the backward pass flow through recon_for_loss (the
                simulated, differentiable reconstruction) -- identical in spirit
                to the straight-through estimator already used for SAIL
                (manuscript Methods §7), applied here to GD's existing,
                non-learned gradient step rather than to a trained model's
                weights. GD's own update rule (Adam on the phase parameter) is
                completely unchanged either way; only the source of the value
                being minimized differs. Default None: behaviour is
                byte-identical to the original published algorithm (recon_for_loss
                used directly, never substituted).
            dc_mask: optional (H,W) or (1,H,W) boolean tensor, True inside the
                zero-order (DC) region to EXCLUDE from the loss. Intended for the
                camera-feedback (CITL) case, where the physical capture contains a
                bright zero-order spike at the DC location that must not be scored
                against the target. When provided, the loss is computed only over
                the kept (non-DC) pixels -- masked pixels contribute nothing to
                the sum and receive no gradient -- exactly matching SAIL+'s
                loss_cam DC exclusion (weight = ~dc_mask), but WITHOUT SAIL+'s
                additional edge weighting (edge_w is intentionally not applied
                here). The mask is static: it is the same every iteration (the DC
                center/radius are fixed for a given optical setup), so it is passed
                once rather than threaded through camera_source per iteration. The
                target is NOT renormalized to exclude DC -- it is a clean image
                with no DC spike, so full-frame normalization is correct, matching
                how SAIL+ leaves its target (I_tgt_E). Default None: full-frame
                MSE, byte-identical to the published algorithm.
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

        A = np.asarray(target_amp, dtype=np.float32)
        if A.ndim != 2:
            raise ValueError(f"Expected target_amp shape (H,W), got {A.shape}")

        H, W = A.shape
        # Recover the intensity training target. Squaring is only correct when A
        # is genuinely an amplitude array (target_formulation="amplitude"); when A
        # is already an intensity array (target_formulation="intensity"), it must
        # be used as-is. See module docstring for why this distinction matters.
        if target_formulation == "amplitude":
            target_intensity_np = A ** 2
        else:  # "intensity"
            target_intensity_np = A

        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        target_intensity = torch.from_numpy(target_intensity_np).to(dev).unsqueeze(0)

        # ---- Padded-resolution target upsampling (FIX) ----
        # hologram_intensity_from_phase (physics.py) returns the replay at
        # PADDED resolution (M,N) = (pad_factor*H, pad_factor*W) when
        # pad_factor > 1 -- it does NOT crop back to (H,W) (see physics.py
        # docstring: this preserves the finer sampling padding buys). The
        # target must therefore be upsampled to the SAME (M,N) before it can
        # be compared to recon, or the loss's subtraction shape-mismatches.
        # The SLM-plane phase itself stays at the ORIGINAL (H,W) throughout --
        # only the target/recon comparison lives at padded resolution.
        if pad_factor > 1:
            M, N = H * pad_factor, W * pad_factor
            target_intensity = torch.nn.functional.interpolate(
                target_intensity.unsqueeze(0), size=(M, N),
                mode="bicubic", align_corners=False,
            ).squeeze(0)
        else:
            M, N = H, W

        # Sum-normalize then rescale by PADDED pixel count (M*N, not H*W) for
        # the training loss. This is an internal training-loss convention
        # only -- it is NOT the same normalization used for final reported
        # metrics (see
        # evaluate_methods.reconstruct_from_phase_np / compute_metrics, which
        # re-propagate the returned phase and re-normalize independently via
        # stats_torch.normalize_intensity_sum before scoring). Decoupling these
        # two normalizations is intentional: it keeps "how the optimizer is
        # driven" separate from "how the result is scored", and was verified by
        # trace to introduce no asymmetry between methods or target_formulations
        # in the final reported PSNR/SSIM/NMSE numbers. Using M*N (rather than
        # H*W) keeps this convention correct at any pad_factor: at pad_factor=1,
        # M*N == H*W, so this is byte-identical to the original scaling.
        target_for_loss = target_intensity / (target_intensity.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (M * N)

        if show:
            plt.figure(figsize=(4, 4))
            plt.imshow(A, cmap="gray")
            plt.title("Target amplitude" if target_formulation == "amplitude" else "Target intensity")
            plt.axis("off")
            plt.show()

        if isinstance(init_phase, np.ndarray):
            if init_phase.shape != (H, W):
                raise ValueError(f"init_phase array shape {init_phase.shape} != expected ({H},{W})")
            init = init_phase.astype(np.float32)
        elif init_phase == "random":
            init = (2 * np.pi * np.random.rand(H, W) - np.pi).astype(np.float32)
        elif init_phase == "zeros":
            init = np.zeros((H, W), dtype=np.float32)
        else:
            raise ValueError(f"Unknown init_phase: {init_phase}")

        phase = torch.nn.Parameter(torch.from_numpy(init).to(dev))

        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([phase], lr=lr)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([phase], lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

        best_loss = float("inf")
        best_phase = None
        best_iteration = None
        loss_history = []

        iter_records = []
        t_run0 = time.perf_counter()

        csv_file = None
        csv_writer = None

        if log_csv_path is not None:
            csv_file = open(log_csv_path, "w", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "iteration",
                    "loss",
                    "best_loss",
                    "is_best",
                    "elapsed_s",
                    "lr",
                ],
            )
            csv_writer.writeheader()

        for k in tqdm(range(iterations), desc="Gradient Descent"):
            optimizer.zero_grad()

            wrapped_phase = torch.atan2(torch.sin(phase), torch.cos(phase))

            if binary_phase:
                quantized = torch.where(
                    torch.cos(wrapped_phase) >= 0,
                    torch.zeros_like(wrapped_phase),
                    torch.full_like(wrapped_phase, np.pi)
                )
                phase_for_forward = quantized
            else:
                phase_for_forward = wrapped_phase

            # Forward propagation through the (shared, manuscript-wide) Fraunhofer
            # model -- see physics.hologram_intensity_from_phase.
            recon = hologram_intensity_from_phase(
                phase_for_forward.unsqueeze(0),
                pad_factor=pad_factor,
                apply_sinc=apply_sinc,
                fill_factor=fill_factor,
                fill_is_areal=fill_is_areal
            )
            recon_for_loss = recon / (recon.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (M * N)

            # Camera-feedback hook: if camera_source is provided, the LOSS VALUE
            # comes from the camera's measured intensity for this iteration's
            # phase, while GRADIENTS still flow through recon_for_loss (the
            # simulated, differentiable reconstruction) -- a straight-through
            # estimator, identical in spirit to the one already used for SAIL
            # (Methods §7), applied here to GD's existing gradient step. GD's
            # own optimizer/update rule is unchanged either way.
            if camera_source is not None:
                I_camera = camera_source(phase_for_forward.detach(), k)
                if I_camera.shape != recon_for_loss.shape:
                    raise ValueError(
                        f"camera_source returned shape {tuple(I_camera.shape)}, "
                        f"expected {tuple(recon_for_loss.shape)}."
                    )
                target_for_value = recon_for_loss + (I_camera - recon_for_loss).detach()
            else:
                target_for_value = recon_for_loss

            # DC-masked loss. When dc_mask is provided, the DC region is
            # excluded from the loss (its pixels contribute nothing to the sum
            # and receive no gradient), matching SAIL+'s loss_cam convention
            # (Methods §8) with weight = (~dc_mask) -- i.e. SAIL+'s DC exclusion
            # WITHOUT its edge weighting (edge_w is deliberately not applied
            # here; this is the bare DC mask only). The target (target_for_loss)
            # is left full-frame sum-normalized, exactly as SAIL+ leaves I_tgt_E:
            # the target is a clean image with no zero-order spike, so only the
            # camera measurement needs DC-excluded normalization (done upstream
            # in load_camera_capture_for_citl), not the target.
            # When dc_mask is None, behaviour is byte-identical to the original
            # published full-frame MSE.
            if dc_mask is not None:
                # dc_mask must be at the SAME (padded, when pad_factor>1)
                # resolution as target_for_value -- (H,W) built with the
                # original DC_CENTER/DC_RADIUS is WRONG once pad_factor>1;
                # center and radius must both be scaled by pad_factor and the
                # mask rebuilt at (M,N). Fail loudly and specifically here
                # rather than let this crash as an opaque broadcast error, or
                # (worse) silently mis-mask if some dims happen to coincide.
                mask_hw = dc_mask.shape[-2:]
                if tuple(mask_hw) != (M, N):
                    raise ValueError(
                        f"dc_mask has shape {tuple(dc_mask.shape)} (last two dims "
                        f"{tuple(mask_hw)}) but target_for_value is at resolution "
                        f"({M},{N}) (pad_factor={pad_factor}). dc_mask must be built "
                        f"at the SAME padded resolution -- rebuild it with "
                        f"DC_CENTER and DC_RADIUS both scaled by pad_factor "
                        f"(e.g. circular_mask_np((M,N), DC_CENTER[0]*pad_factor, "
                        f"DC_CENTER[1]*pad_factor, DC_RADIUS*pad_factor))."
                    )
                keep = (~dc_mask).to(dtype=target_for_value.dtype, device=target_for_value.device)
                while keep.ndim < target_for_value.ndim:
                    keep = keep.unsqueeze(0)
                loss = (keep * (target_for_value - target_for_loss) ** 2).sum() / (keep.sum() + 1e-8)
            else:
                loss = torch.mean((target_for_value - target_for_loss) ** 2)

            if tv_weight > 0.0:
                dy = wrapped_phase[1:, :] - wrapped_phase[:-1, :]
                dx = wrapped_phase[:, 1:] - wrapped_phase[:, :-1]
                tv = dx.abs().mean() + dy.abs().mean()
                loss = loss + tv_weight * tv

            loss.backward()
            optimizer.step()

            current_loss = float(loss.item())
            is_best = current_loss < best_loss

            if is_best:
                best_loss = current_loss
                best_phase = wrapped_phase.detach().clone()
                best_iteration = k

            # NOTE: loss_history was previously initialized but never appended
            # to (a pre-existing bug, unrelated to the pad/DC fixes above) --
            # fixed here so the returned loss_history is actually populated.
            loss_history.append(current_loss)

            record = {
                "iteration": k,
                "loss": current_loss,
                "best_loss": float(best_loss),
                "is_best": bool(is_best),
                "elapsed_s": time.perf_counter() - t_run0,
                "lr": optimizer.param_groups[0]["lr"],
            }

            iter_records.append(record)

            if csv_writer is not None:
                csv_writer.writerow(record)
                csv_file.flush()

            if iteration_callback is not None:
                iteration_callback(record)

        if best_phase is None:
            best_phase = torch.atan2(torch.sin(phase), torch.cos(phase)).detach()
            # Loss never improved from its initial inf value (iterations=0, or
            # a pathological first step) -- best_iteration is undefined in this
            # case since no iteration was ever recorded as "best."
            best_iteration = None

        if lowpass:
            best_phase = phase_lowpass_filter_torch(best_phase, sigma=lpf_sigma)

        phase_np = best_phase.detach().cpu().numpy().astype(np.float32)

        if add_carrier:
            phase_np = self.add_offaxis_carrier(phase_np, fx=carrierX, fy=carrierY)

        phase_np = np.angle(np.exp(1j * phase_np)).astype(np.float32)
        phase_8bit = phase_to_uint8(phase_np)

        if csv_file is not None:
            csv_file.close()

        return {
            "phase_radians": phase_np,
            "phase_uint8": phase_8bit,
            "best_loss": float(best_loss),
            "best_iteration": best_iteration,
            "loss_history": loss_history,
            "iteration_records": iter_records,
        }



def _demo_target_path():
    """First target image in the deposit, for the __main__ demo."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import paths
    pngs = sorted(paths.TARGETS.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no target images in {paths.TARGETS}")
    return str(pngs[0])

if __name__ == "__main__":
    import time
    from PIL import Image

    img = Image.open(
        _demo_target_path()  # <deposit>/targets, see paths.py
    ).convert("L")
    img = np.asarray(img, dtype=np.float32) / 255.0

    # Input image is intensity
    target_intensity = img
    target_amplitude = np.sqrt(target_intensity)

    gd = GradientDescentHologram()

    pad_factor = 1
    apply_sinc = False
    fill_factor = 1
    fill_is_areal = True

    t0 = time.time()
    out = gd.run(
        target_amplitude,
        iterations=200,
        init_phase="random",
        lr=0.03,
        optimizer_name="adam",
        loss_type="mse_sum",
        tv_weight=0.0,
        device="cuda",
        show=True,
        pad_factor=pad_factor,
        apply_sinc=apply_sinc,
        fill_factor=fill_factor,
        fill_is_areal=fill_is_areal
    )
    elapsed = time.time() - t0

    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Best loss: {out['best_loss']:.6e}")

    # Recompute reconstruction from best phase
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_t = torch.from_numpy(out["phase_radians"]).float().to(dev).unsqueeze(0)

    with torch.no_grad():
        recon = hologram_intensity_from_phase(
            phase_t,
            pad_factor=pad_factor,
            apply_sinc=apply_sinc,
            fill_factor=fill_factor,
            fill_is_areal=fill_is_areal
        ).squeeze(0).detach().cpu().numpy()

    # Match the training normalization for fair comparison
    target_norm = target_intensity / (target_intensity.sum() + 1e-12) * target_intensity.size
    recon_norm = recon / (recon.sum() + 1e-12) * recon.size

    # Display versions
    target_view = target_norm - target_norm.min()
    target_view /= (target_view.max() + 1e-12)

    recon_view = recon_norm - recon_norm.min()
    recon_view /= (recon_view.max() + 1e-12)

    error = np.abs(recon_norm - target_norm)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(target_view, cmap="gray")
    plt.title("Target intensity")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(recon_view, cmap="gray")
    plt.title("Reconstructed intensity")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(out["phase_radians"], cmap="twilight")
    plt.title("Phase")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(error, cmap="hot")
    plt.title("Absolute error")
    plt.axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()
