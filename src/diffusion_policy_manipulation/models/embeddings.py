"""
Sinusoidal timestep embeddings.

sinusoidal_timestep_embedding
    Maps a batch of integer diffusion timesteps to continuous float32 vectors
    using the same sin/cos scheme as transformer positional encodings.
    These embeddings are conditioning signals for the denoiser MLP.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Compute sinusoidal embeddings for a batch of diffusion timesteps.

    The embedding for position p at dimension 2i / 2i+1 is:

        emb[p, 2i]   = sin(p / 10000^(2i / dim))
        emb[p, 2i+1] = cos(p / 10000^(2i / dim))

    Parameters
    ----------
    timesteps:
        Integer timestep indices, shape ``[B]``.  Must be int64.
    dim:
        Desired embedding dimension.  If odd, the last dimension is zero-padded.

    Returns
    -------
    torch.Tensor
        Float32 embedding tensor of shape ``[B, dim]``.
    """
    assert timesteps.dim() == 1, "timesteps must be 1-D"

    half = dim // 2
    # Frequencies: 1 / 10000^(k / (half-1)) for k in [0, half)
    # Guard against half==1 to avoid division by zero in log.
    if half > 1:
        log_scale = math.log(10000.0) / (half - 1)
    else:
        log_scale = 0.0

    freqs = torch.exp(
        -log_scale * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    )  # [half]

    # Outer product: [B, 1] * [1, half] -> [B, half]
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim] or [B, dim-1]

    # Pad to exact dim if dim is odd.
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    return emb.to(torch.float32)
