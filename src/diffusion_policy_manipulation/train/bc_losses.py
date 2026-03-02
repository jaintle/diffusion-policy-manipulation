"""
Behavior Cloning loss functions.

gaussian_nll
    Diagonal Gaussian negative log-likelihood used to train the BC policy.
    The loss is averaged over both the batch and action dimensions so it is
    independent of batch size and act_dim, which simplifies learning-rate
    tuning.
"""

from __future__ import annotations

import torch

# log_std clamping range — prevents numerical explosion during early training.
_LOG_STD_MIN: float = -20.0
_LOG_STD_MAX: float = 2.0


def gaussian_nll(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood.

    For a diagonal Gaussian N(mean, diag(std^2)) the per-element NLL is:

        nll_i = log_std_i + 0.5 * log(2π) + 0.5 * ((a_i - μ_i) / std_i)²

    Parameters
    ----------
    mean:
        Predicted action mean, shape ``[batch, act_dim]``.
    log_std:
        Predicted (or learned) log standard deviation, same shape as *mean*.
        Will be clamped to ``[_LOG_STD_MIN, _LOG_STD_MAX]`` for stability.
    actions:
        Ground-truth actions, same shape as *mean*.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over batch × act_dim elements).
    """
    log_std_clamped = log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
    std = log_std_clamped.exp()

    # Squared normalised residual: ((a - μ) / σ)²
    sq_residual = ((actions - mean) / std) ** 2

    # Per-element NLL (dropping the constant 0.5·log(2π) would bias the loss
    # but not affect optimisation; we include it for correctness).
    nll_elementwise = log_std_clamped + 0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi, device=mean.device)) + sq_residual
    )

    return nll_elementwise.mean()
