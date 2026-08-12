"""HALO - Holographic Attention-based Learned Operator.

The hologram generator of the paper. Earlier internal versions used the
working name HoT; recorded runs keep the identifiers they were launched
with, and checkpoints load unchanged because torch stores state-dict keys,
not class names.
"""
import torch
import torch.nn as nn

class HALO(nn.Module):
    def __init__(
        self,
        H: int = 28,
        W: int = 28,
        p: int = 14,
        d_model: int = 256,
        nhead: int = 16,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        pre_norm: bool = True,
        output_mode: str = "patch",   # "patch" or "global_field"
        H_out: int = 14,
        W_out: int = 14,
        in_channels=1,
        out_channels=2,
    ):
        super().__init__()

        if H % p != 0 or W % p != 0:
            raise ValueError(f"H and W must be divisible by p. Got H={H}, W={W}, p={p}")

        if output_mode not in {"patch", "global_field"}:
            raise ValueError(f"output_mode must be 'patch' or 'global_field', got {output_mode}")

        self.H = H
        self.W = W
        self.p = p
        self.h = H // p
        self.w = W // p
        self.T = self.h * self.w

        self.output_mode = output_mode
        self.H_out = H_out
        self.W_out = W_out

        self.in_channels = in_channels
        self.out_channels = out_channels
        in_dim = self.in_channels * p * p
        self.patch_embed = nn.Linear(in_dim, d_model)

        self.pos2d = nn.Parameter(torch.zeros(1, self.h, self.w, d_model))
        nn.init.trunc_normal_(self.pos2d, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=pre_norm,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if output_mode == "patch":
            self.head = nn.Linear(d_model, self.out_channels * p * p)
        else:
            self.head = nn.Linear(d_model, self.out_channels * H_out * W_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        patch mode:
            input  X: (B, T, p*p)
            output y: (B, T, 2*p*p)

        global_field mode:
            input  X: (B, T, p*p)
            output Y: (B, 2, H_out, W_out)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X shape (B,T,D), got {X.shape}")
        if X.shape[1] != self.T:
            raise ValueError(f"Expected T={self.T}, got {X.shape[1]}")
        if X.shape[2] != self.in_channels * self.p * self.p:
            raise ValueError(f"Expected D={self.in_channels*self.p*self.p}, got {X.shape[2]}")

        B = X.shape[0]

        z = self.patch_embed(X)                          # (B,T,d)
        pos = self.pos2d.view(1, self.T, z.shape[-1])
        z = z + pos
        z = self.encoder(z)                              # (B,T,d)

        if self.output_mode == "patch":
            y = self.head(z)                             # (B,T,2*p*p)
            return y

        z_global = z.mean(dim=1)                         # (B,d)
        y = self.head(z_global)                          # (B,2*H_out*W_out)
        Y = y.view(B, self.out_channels, self.H_out, self.W_out)        # (B,2,H_out,W_out)
        return Y
