r"""
sail_simulation_demo.py - the method end to end, no hardware required.

This trains the HALO generator against the differentiable Fraunhofer forward
model on a single target and reports the reconstruction quality as it
improves. It is the simulation half of SAIL. The camera half replaces the
simulated intensity with a measured one through the straight-through
estimator, and that half needs the bench, so this demonstration is the part
a reader can run anywhere.

It runs on CPU. The full-size model is the one from the paper, four
half-field tokens at p = 500 over a 1000x1000 grid, and a CPU epoch takes a
few seconds. The default budget is small so the script finishes in minutes
and shows the quality climbing. Give it more epochs for a better hologram.

    python sail_simulation_demo.py                 # defaults
    python sail_simulation_demo.py --epochs 500    # longer, better
    python sail_simulation_demo.py --target alley  # a specific deposit target

If the deposit's targets directory is present, the demo trains on a real
target image. A bare clone without the data synthesizes a test pattern
instead, so the script runs either way.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from HALO import HALO
from patching import patchify, unpatchify
from physics import hologram_intensity_from_field
from stats_torch import normalize_intensity_sum

try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
except ImportError:
    _sk_psnr = None


def load_target(name: str | None, size: int) -> tuple[torch.Tensor, str]:
    """A target intensity in [0, 1], from the deposit if available."""
    try:
        import paths
        pngs = sorted(paths.TARGETS.glob("*.png"))
    except Exception:
        pngs = []
    if pngs:
        if name is not None:
            match = [p for p in pngs if p.stem == name]
            if not match:
                raise SystemExit(
                    f"target {name!r} not found. Available targets are "
                    + ", ".join(p.stem for p in pngs))
            chosen = match[0]
        else:
            chosen = pngs[0]
        from PIL import Image
        img = Image.open(chosen).convert("L").resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr), chosen.stem
    # No data present. Synthesize a pattern with structure at several scales
    # so the reconstruction is visually meaningful.
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size),
                         indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2)
    rings = 0.5 + 0.5 * np.cos(18.0 * np.pi * r)
    bars = (np.sin(40.0 * np.pi * xx) > 0).astype(np.float32)
    quad = ((xx > 0) ^ (yy > 0)).astype(np.float32)
    arr = (0.5 * rings + 0.3 * bars + 0.2 * quad).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return torch.from_numpy(arr), "synthetic test pattern"


def psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR with the same convention as the paper's scoring, skimage with
    data_range 1.0 on the energy-normalized reconstruction against the
    target in [0, 1]."""
    r = recon.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    if _sk_psnr is not None:
        return float(_sk_psnr(t, r, data_range=1.0))
    mse = float(np.mean((t - r) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--size", type=int, default=1000,
                    help="grid size. 1000 is the paper configuration.")
    ap.add_argument("--patch", type=int, default=None,
                    help="patch size p. Defaults to size divided by 2, the "
                         "half-field tokens of the paper.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target", type=str, default=None,
                    help="a target stem from the deposit's targets directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = args.patch if args.patch is not None else args.size // 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target, target_name = load_target(args.target, args.size)
    target = target.to(device)
    # The model sees the raw target in [0, 1]. The loss compares
    # energy-normalized intensities scaled to mean brightness one, the same
    # convention as the recorded training runs.
    scale = float(args.size * args.size)
    I_tgt = normalize_intensity_sum(target.unsqueeze(0)) * scale

    model = HALO(H=args.size, W=args.size, p=p, d_model=256, nhead=16,
                 num_layers=4, dim_feedforward=1024, in_channels=1,
                 out_channels=2, output_mode="patch").to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"target          {target_name}")
    print(f"grid            {args.size}x{args.size}, p = {p}, "
          f"{(args.size // p) ** 2} tokens")
    print(f"parameters      {n_params:,}")
    print(f"device          {device}")
    print(f"epochs          {args.epochs}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    X = patchify(target.unsqueeze(0), p)
    t0 = time.time()
    report_every = max(1, args.epochs // 10)
    for epoch in range(args.epochs):
        opt.zero_grad()
        tokens = model(X)
        Y = unpatchify(tokens, args.size, args.size, p, C=2)
        I_sim = normalize_intensity_sum(
            hologram_intensity_from_field(Y)) * scale
        loss = torch.mean((I_sim - I_tgt) ** 2)
        loss.backward()
        opt.step()
        if epoch % report_every == 0 or epoch == args.epochs - 1:
            q = psnr(I_sim[0], target)
            print(f"epoch {epoch:5d}   loss {loss.item():.3e}   "
                  f"PSNR {q:6.2f} dB   {time.time() - t0:6.1f} s")

    q = psnr(I_sim[0], target)
    print(f"\nfinished. Final PSNR {q:.2f} dB after {args.epochs} epochs "
          f"({time.time() - t0:.1f} s).")
    print("The phase hologram is torch.atan2 of the two field channels. "
          "The paper's per-target simulation runs use this same loop for "
          "10,000 epochs, and the camera-in-the-loop runs replace I_sim "
          "with the measured intensity through the straight-through "
          "estimator in sail.ipynb.")


if __name__ == "__main__":
    main()
