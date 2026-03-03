"""
Diffusion training loss.

diffusion_eps_loss
    Standard epsilon-prediction loss used to train the MLP denoiser.
    Given a ground-truth action sequence, it:
      1. Samples a noisy version x_t using the forward diffusion process.
      2. Asks the denoiser to predict the added noise (eps-prediction target).
      3. Returns the mean squared error between prediction and ground truth.

This is the only loss variant used in this project (no v-prediction, no x0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusion_policy_manipulation.models.diffusion_schedule import DiffusionSchedule


def diffusion_eps_loss(
    denoiser: torch.nn.Module,
    schedule: DiffusionSchedule,
    obs: torch.Tensor,
    actions: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Epsilon-prediction MSE loss for diffusion policy training.

    Forward process:
        x_t = sqrt(ᾱ_t) · actions  +  sqrt(1 - ᾱ_t) · noise

    Training objective:
        L = MSE( denoiser(obs, x_t, t),  noise )

    Parameters
    ----------
    denoiser:
        MLPDenoiser (or compatible module).
    schedule:
        Pre-computed DiffusionSchedule; ``alpha_bars`` must be on the same
        device as *obs*.
    obs:
        Conditioning observations, shape ``[B, obs_dim]``.
    actions:
        Ground-truth action sequences, shape ``[B, H, act_dim]``.
    t:
        Diffusion timestep indices, shape ``[B]`` int64.
    noise:
        Gaussian noise drawn from N(0, I), shape ``[B, H, act_dim]``.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss averaged over batch × H × act_dim elements.
    """
    # alpha_bar_t: [B] → [B, 1, 1] for broadcasting with [B, H, act_dim]
    alpha_bar_t = schedule.alpha_bars[t].view(-1, 1, 1)

    # Forward diffusion: corrupt ground-truth actions with noise.
    x_t = alpha_bar_t.sqrt() * actions + (1.0 - alpha_bar_t).sqrt() * noise

    # Predict the noise that was added.
    eps_hat = denoiser(obs, x_t, t)  # [B, H, act_dim]

    return F.mse_loss(eps_hat, noise)
