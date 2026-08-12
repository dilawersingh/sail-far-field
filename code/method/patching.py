import torch

def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    x: (B, H, W) OR (B, C, H, W)

    returns:
        (B, T, C*p*p)  if input has channels
        (B, T, p*p)    if single channel
    """

    if x.ndim == 3:
        x = x.unsqueeze(1)   # (B,1,H,W)

    if x.ndim != 4:
        raise ValueError(f"patchify expected (B,H,W) or (B,C,H,W), got {x.shape}")

    B, C, H, W = x.shape

    if H % p != 0 or W % p != 0:
        raise ValueError(f"H and W must be divisible by p. Got H={H}, W={W}, p={p}")

    h = H // p
    w = W // p

    x = x.view(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    tokens = x.view(B, h*w, C*p*p)

    return tokens


def unpatchify(tokens: torch.Tensor, H: int, W: int, p: int, C: int) -> torch.Tensor:
    """
    tokens: (B, T, C*p*p), where T=(H//p)*(W//p)
    returns: (B, C, H, W)
    """
    if tokens.ndim != 3:
        raise ValueError(f"unpatchify expected tokens.ndim=3 (B,T,D), got {tokens.shape}")

    B, T, D = tokens.shape
    if H % p != 0 or W % p != 0:
        raise ValueError(f"H and W must be divisible by p. Got H={H}, W={W}, p={p}")

    h = H // p
    w = W // p
    expected_T = h * w
    expected_D = C * p * p

    if T != expected_T:
        raise ValueError(f"Expected T={expected_T} (from H,W,p), got T={T}")
    if D != expected_D:
        raise ValueError(f"Expected D={expected_D} (=C*p*p), got D={D}")

    # (B, T, C*p*p) -> (B, h, w, C, p, p)
    x = tokens.view(B, h, w, C, p, p)
    # (B, C, h, p, w, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    # (B, C, H, W)
    out = x.view(B, C, H, W)
    return out