"""
Gaussian MLP Behavior Cloning policy.

Architecture
------------
- MLP backbone: [obs_dim] -> hidden_dim x num_layers -> [act_dim]  (mean head)
- Learned log_std parameter vector of shape [act_dim] (option A — fixed per
  action dimension, not input-dependent).  This keeps the model minimal while
  still producing a proper Gaussian distribution for NLL training.

Usage
-----
::

    policy = GaussianMLPPolicy(obs_dim=5, act_dim=2, hidden_dim=256, num_layers=2)
    mean, log_std = policy(obs_tensor)          # forward pass
    action = policy.deterministic_action(obs)   # mean only, for eval
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianMLPPolicy(nn.Module):
    """MLP that outputs a diagonal Gaussian over actions.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    act_dim:
        Dimensionality of the action vector.
    hidden_dim:
        Width of each hidden layer.
    num_layers:
        Number of hidden layers (minimum 1).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        # Build MLP backbone: input -> hidden x num_layers -> mean output
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, act_dim))  # mean head

        self.backbone = nn.Sequential(*layers)

        # Learned log_std: one scalar per action dimension, input-independent.
        # Initialised to 0 (std ≈ 1) so early training is well-scaled.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std for the action distribution.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``[batch, obs_dim]``.

        Returns
        -------
        mean : torch.Tensor
            Shape ``[batch, act_dim]``.
        log_std : torch.Tensor
            Shape ``[batch, act_dim]`` (log_std parameter broadcast over batch).
        """
        mean = self.backbone(obs)
        # Broadcast the parameter [act_dim] -> [batch, act_dim]
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return mean, log_std

    # ------------------------------------------------------------------
    # Deterministic action (evaluation)
    # ------------------------------------------------------------------

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action (no sampling).

        Used exclusively during evaluation so outputs are reproducible.

        Parameters
        ----------
        obs:
            Observation tensor of shape ``[batch, obs_dim]`` or ``[obs_dim]``.

        Returns
        -------
        torch.Tensor
            Mean action, same leading shape as *obs*.
        """
        mean, _ = self.forward(obs)
        return mean
