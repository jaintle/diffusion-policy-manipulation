"""
Deterministic DDIM sampler for action-sequence diffusion policies.

sample_ddim
    Generates a denoised action sequence from Gaussian noise using the DDIM
    update rule with eta=0 (fully deterministic).

DDIM update (eta=0)
-------------------
Given the noisy sequence x_t at step t and the predicted noise eps:

    x0_pred  = (x_t  - sqrt(1 - ᾱ_t) * eps) / sqrt(ᾱ_t)
    x_{prev} = sqrt(ᾱ_{prev}) * x0_pred  +  sqrt(1 - ᾱ_{prev}) * eps

At the last denoising step (no previous timestep) x0_pred is returned
directly.

Reference: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021).
"""

from __future__ import annotations

import torch

from diffusion_policy_manipulation.models.diffusion_schedule import DiffusionSchedule


@torch.no_grad()
def sample_ddim(
    denoiser: torch.nn.Module,
    schedule: DiffusionSchedule,
    obs: torch.Tensor,
    horizon: int,
    act_dim: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Draw a denoised action sequence using deterministic DDIM.

    Parameters
    ----------
    denoiser:
        MLPDenoiser (or compatible module) that accepts ``(obs, x_t, t)`` and
        returns predicted noise of shape ``[B, horizon, act_dim]``.
    schedule:
        Pre-computed DiffusionSchedule with alpha_bars of length T.
    obs:
        Observation conditioning tensor, shape ``[B, obs_dim]``.
    horizon:
        Action sequence length H.
    act_dim:
        Action dimensionality.
    steps:
        Number of DDIM denoising steps (≤ T).  Evenly spaced timestep
        indices are selected from ``[0, T-1]``.
    seed:
        Seed for the initial noise generator.  The same seed always
        produces the same output, enabling deterministic evaluation.

    Returns
    -------
    torch.Tensor
        Denoised action sequence of shape ``[B, horizon, act_dim]`` float32.
    """
    B = obs.shape[0]
    device = obs.device
    T = schedule.alpha_bars.shape[0]

    # ------------------------------------------------------------------
    # Initial noise — seeded deterministically.
    # A new Generator is created on every call so seed isolation is total:
    # the caller's global RNG state is never touched.
    # ------------------------------------------------------------------
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x_t = torch.randn(B, horizon, act_dim, generator=gen, device=device)

    # ------------------------------------------------------------------
    # Build the timestep index sequence.
    # linspace gives steps evenly spaced floats in [0, T-1]; .long()
    # truncates to integers.  We iterate largest → smallest.
    # ------------------------------------------------------------------
    indices = torch.linspace(0, T - 1, steps, device=device).long()  # [steps]
    indices = indices.flip(0)                                         # largest first

    # ------------------------------------------------------------------
    # Reverse diffusion loop.
    # ------------------------------------------------------------------
    for i in range(steps):
        t_idx = indices[i]                                    # scalar int64 tensor
        t_batch = t_idx.expand(B)                             # [B] (same t for all)

        alpha_bar_t = schedule.alpha_bars[t_idx]              # scalar

        # Predict noise at current step.
        eps = denoiser(obs, x_t, t_batch)                     # [B, horizon, act_dim]

        # Reconstruct x0 prediction.
        x0_pred = (x_t - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()

        if i < steps - 1:
            # DDIM update: step towards previous (smaller) timestep.
            t_prev_idx = indices[i + 1]
            alpha_bar_prev = schedule.alpha_bars[t_prev_idx]  # scalar
            x_t = alpha_bar_prev.sqrt() * x0_pred + (1.0 - alpha_bar_prev).sqrt() * eps
        else:
            # Final denoising step — return x0 prediction directly.
            x_t = x0_pred

    return x_t.to(torch.float32)
