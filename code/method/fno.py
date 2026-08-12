"""
FNO.py

Fourier Neural Operator (2-D), written as a drop-in replacement for HoT in the
hologram-generation pipeline.

Interface contract (this is the whole point of the file)
--------------------------------------------------------
HoT path:   xb (B,H,W) -> patchify -> HoT -> unpatchify(..., C=2) -> Y (B,2,H,W)
FNO path:   xb (B,H,W) ---------------> FNO2d ------------------> Y (B,2,H,W)

Everything downstream of Y (field_to_phase, hologram_intensity_from_field,
the straight-through estimator, phase_to_slm_frame, the capture pipeline) is
byte-identical between the two. FNO does NOT patchify: the FFT is O(n log n),
so it consumes the full HxW grid directly and its learned parameters live in
Fourier-mode space rather than pixel space.

Architecture
------------
Each FNO layer runs two paths in parallel and sums them:
  1. SPECTRAL: rfft2 -> keep the lowest `modes1 x modes2` Fourier modes ->
     multiply each retained mode by a learned complex weight -> irfft2.
     This is a GLOBAL operation (every output pixel depends on every input
     pixel), but the mixing is diagonal in the Fourier basis: mode k in maps
     to mode k out. The basis is fixed by construction; only the per-mode
     scaling is learned.
  2. POINTWISE: a 1x1 convolution, giving a purely local path that carries
     high-frequency content the truncated spectral path discards.
Sum, apply nonlinearity, repeat. A lifting 1x1 conv precedes the stack and a
two-layer 1x1 projection follows it.

This fixed-basis property is exactly the architectural point of comparison
against self-attention, which learns the coupling pattern itself rather than
inheriting it from the Fourier basis.

Notes on choices made here
--------------------------
- Complex weights are STORED as real tensors with a trailing dim of size 2 and
  viewed as complex at use time (torch.view_as_complex). Native complex
  nn.Parameters have historically had rough edges with optimizer states and
  weight decay; this formulation sidesteps all of that and is numerically
  identical.
- Initialization follows the reference FNO implementation (Li et al.):
  scale = 1/(in_channels*out_channels), weights ~ scale * U[0,1). Kept
  deliberately close to the canonical implementation so the baseline cannot be
  accused of being a bespoke variant.
- FFTs use norm="ortho" so the forward/inverse pair is unitary and the
  effective weight scale does not depend on H, W.
- Mode counts are validated against H, W at every forward pass and raise
  rather than silently truncating.

Memory
------
Activations are (B, width, H, W) at every layer, and the spectral path
allocates two complex tensors of shape (B, width, H, W//2+1) per layer. At
H=W=1000 and width=32 that is ~128 MB per tensor in float32; budget roughly
2.5-3 GB of activations for a 4-layer model plus parameters and Adam states.
width=64 roughly doubles activations and quadruples spectral parameters.
Use fno_param_count() to check parameter budget before instantiating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _resolve_activation(name):
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. Choices: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]()


class SpectralConv2d(nn.Module):
    """One spectral convolution: truncated Fourier multiplication.

    Input  (B, in_channels,  H, W)
    Output (B, out_channels, H, W)

    rfft2 halves the last axis (W -> W//2+1) but keeps the full range on the
    second-to-last axis (H), where negative frequencies live at the end. Hence
    TWO weight tensors: `weights_pos` for the lowest `modes1` positive-frequency
    rows and `weights_neg` for the lowest `modes1` negative-frequency rows.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        if modes1 < 1 or modes2 < 1:
            raise ValueError(f"modes must be >= 1, got modes1={modes1}, modes2={modes2}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # trailing dim of 2 == (real, imag); viewed as complex in forward()
        self.weights_pos = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

    @staticmethod
    def _spectral_matmul(x_ft, weight_real):
        w = torch.view_as_complex(weight_real.contiguous())
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        if self.modes1 > H // 2:
            raise ValueError(
                f"modes1={self.modes1} exceeds H//2={H // 2} for H={H}. The positive- "
                f"and negative-frequency mode blocks would overlap."
            )
        if self.modes2 > W // 2 + 1:
            raise ValueError(
                f"modes2={self.modes2} exceeds W//2+1={W // 2 + 1} for W={W}."
            )

        x_ft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1) complex

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=x_ft.dtype, device=x.device
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = self._spectral_matmul(x_ft[:, :, :m1, :m2], self.weights_pos)
        out_ft[:, :, -m1:, :m2] = self._spectral_matmul(x_ft[:, :, -m1:, :m2], self.weights_neg)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    """Fourier Neural Operator producing a 2-channel (real, imag) field.

    forward:
        input  x: (B, H, W) or (B, in_channels, H, W)
        output Y: (B, out_channels, H, W)   -- out_channels=2 => (re, im)

    The output is deliberately unconstrained in scale: only its argument
    matters downstream (field_to_phase computes atan2(im, re)).
    """

    def __init__(
        self,
        H: int,
        W: int,
        modes1: int,
        modes2: int,
        width: int,
        num_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 2,
        include_grid: bool = True,
        input_scale: float = 1.0,
        projection_width: int = 128,
        activation: str = "gelu",
        domain_padding: int = 0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if domain_padding < 0:
            raise ValueError(f"domain_padding must be >= 0, got {domain_padding}")

        # Modes are validated against the PADDED grid, since that is what the
        # spectral layers actually see.
        H_eff, W_eff = H + domain_padding, W + domain_padding
        if modes1 > H_eff // 2:
            raise ValueError(f"modes1={modes1} exceeds H_eff//2={H_eff // 2} (H={H}, pad={domain_padding})")
        if modes2 > W_eff // 2 + 1:
            raise ValueError(f"modes2={modes2} exceeds W_eff//2+1={W_eff // 2 + 1} (W={W}, pad={domain_padding})")

        self.H = H
        self.W = W
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_grid = include_grid
        self.input_scale = float(input_scale)
        self.projection_width = projection_width
        self.activation_name = activation
        self.domain_padding = domain_padding

        lift_in = in_channels + (2 if include_grid else 0)
        self.lift = nn.Conv2d(lift_in, width, kernel_size=1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)]
        )
        self.act = _resolve_activation(activation)

        self.project_1 = nn.Conv2d(width, projection_width, kernel_size=1)
        self.project_2 = nn.Conv2d(projection_width, out_channels, kernel_size=1)

        if include_grid:
            # normalized coordinate channels, cached once
            ys = torch.linspace(0.0, 1.0, H).view(1, 1, H, 1).expand(1, 1, H, W)
            xs = torch.linspace(0.0, 1.0, W).view(1, 1, 1, W).expand(1, 1, H, W)
            self.register_buffer("grid", torch.cat([ys, xs], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected (B,H,W) or (B,C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Expected spatial size ({self.H},{self.W}), got ({H},{W})")

        x = x * self.input_scale

        if self.include_grid:
            x = torch.cat([x, self.grid.expand(B, -1, -1, -1)], dim=1)

        x = self.lift(x)

        # Domain padding, following the reference FNO implementation for
        # NON-PERIODIC inputs. rfft2 treats its input as periodic, so without
        # this the spectral convolution wraps content from one edge of the
        # image around to the other. The reference omits it for genuinely
        # periodic problems (Navier-Stokes on a torus) and includes it for
        # non-periodic ones (Darcy flow); target photographs are firmly the
        # latter. Padded after lifting and cropped before projection, so the
        # extra region only ever exists inside the layer stack.
        pad = self.domain_padding
        if pad > 0:
            x = F.pad(x, [0, pad, 0, pad])

        last = self.num_layers - 1
        for i, (spec, pw) in enumerate(zip(self.spectral_layers, self.pointwise_layers)):
            x = spec(x) + pw(x)
            if i != last:
                x = self.act(x)

        if pad > 0:
            x = x[..., :-pad, :-pad]

        x = self.project_1(x)
        x = self.act(x)
        x = self.project_2(x)
        return x  # (B, out_channels, H, W)


def fno_param_count(
    width: int,
    modes1: int,
    modes2: int,
    num_layers: int = 4,
    in_channels: int = 1,
    out_channels: int = 2,
    include_grid: bool = True,
    projection_width: int = 128,
):
    """Exact trainable-parameter count without instantiating the model.

    Useful for picking a configuration against a parameter budget before
    allocating anything on the GPU. Returns (total, breakdown_dict).
    """
    lift_in = in_channels + (2 if include_grid else 0)
    lift = lift_in * width + width

    # complex weights stored as 2 reals; 2 weight tensors (pos/neg freq) per layer
    spectral_per_layer = 2 * (width * width * modes1 * modes2) * 2
    pointwise_per_layer = width * width + width
    layers = num_layers * (spectral_per_layer + pointwise_per_layer)

    proj1 = width * projection_width + projection_width
    proj2 = projection_width * out_channels + out_channels

    breakdown = {
        "lift": lift,
        "spectral_total": num_layers * spectral_per_layer,
        "pointwise_total": num_layers * pointwise_per_layer,
        "project_1": proj1,
        "project_2": proj2,
    }
    total = lift + layers + proj1 + proj2
    return total, breakdown


if __name__ == "__main__":
    # Parameter budget reference for H=W=1000, 4 layers.
    # HoT (SAIL) at p=500, d_model=256 is 195,660,320 trainable parameters.
    print(f"{'width':>6} {'modes':>6} {'params':>15}")
    for width in (32, 64):
        for modes in (32, 64, 96, 109, 128):
            total, _ = fno_param_count(width, modes, modes, num_layers=4)
            print(f"{width:>6} {modes:>6} {total:>15,}")