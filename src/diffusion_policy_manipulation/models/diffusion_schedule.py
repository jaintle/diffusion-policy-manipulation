"""
Linear beta diffusion schedule.

DiffusionSchedule
    Dataclass holding the pre-computed scalar tensors needed at every
    step of the forward/reverse diffusion process.

make_linear_schedule
    Constructs a DiffusionSchedule with betas evenly spaced between
    beta_start and beta_end over T steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    """Pre-computed diffusion schedule tensors, all of shape [T].

    Attributes
    ----------
    betas : torch.Tensor
        Noise variances at each diffusion step.  Shape ``[T]``.
    alphas : torch.Tensor
        ``1 - betas``.  Shape ``[T]``.
    alpha_bars : torch.Tensor
        Cumulative product of alphas: ``cumprod(alphas)``.  Shape ``[T]``.
        ``alpha_bars[t]`` corresponds to ``ᾱ_t`` in the DDPM/DDIM literature.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_linear_schedule(
    T: int,
    beta_start: float,
    beta_end: float,
    device: str,
) -> DiffusionSchedule:
    """Build a linear beta schedule.

    Parameters
    ----------
    T:
        Total number of diffusion timesteps.
    beta_start:
        Value of beta at the first timestep (t=0).
    beta_end:
        Value of beta at the last timestep (t=T-1).
    device:
        Torch device string for all returned tensors.

    Returns
    -------
    DiffusionSchedule
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)
