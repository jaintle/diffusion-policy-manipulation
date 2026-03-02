"""
MLP denoiser for diffusion-based action-sequence modeling.

MLPDenoiser
    Predicts the noise eps_hat added at diffusion step t, conditioned on a
    flat observation vector and the noisy action sequence x_t.

    Input:
        obs   [B, obs_dim]             – observation conditioning
        x_t   [B, horizon, act_dim]    – noisy action sequence at step t
        t     [B] int64                – diffusion timestep indices

    Output:
        eps_hat  [B, horizon, act_dim] – predicted noise

Architecture
------------
    1. t is embedded via sinusoidal_timestep_embedding -> [B, t_embed_dim]
       then projected through a Linear + ReLU to hidden_dim.
    2. x_t is flattened to [B, horizon * act_dim].
    3. [obs, x_flat, t_proj] are concatenated and fed through a num_layers-deep
       ReLU MLP of width hidden_dim.
    4. The final linear head produces [B, horizon * act_dim], reshaped to
       [B, horizon, act_dim].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_policy_manipulation.models.embeddings import sinusoidal_timestep_embedding


class MLPDenoiser(nn.Module):
    """Minimal MLP noise predictor for DDIM sampling.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    act_dim:
        Dimensionality of a single action step.
    horizon:
        Number of action steps in the predicted sequence (H).
    hidden_dim:
        Width of each hidden layer.
    num_layers:
        Number of hidden layers (minimum 1).
    t_embed_dim:
        Sinusoidal embedding dimension for the diffusion timestep.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
        t_embed_dim: int,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.horizon = horizon
        self.act_dim = act_dim
        self.t_embed_dim = t_embed_dim

        # Project sinusoidal t embedding -> hidden_dim conditioning vector.
        self.t_proj = nn.Linear(t_embed_dim, hidden_dim)

        # MLP body: concatenated inputs -> hidden -> ... -> noise prediction.
        # Input size = obs_dim + (horizon * act_dim) + hidden_dim
        in_dim = obs_dim + horizon * act_dim + hidden_dim

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, horizon * act_dim))

        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise component of a noisy action sequence.

        Parameters
        ----------
        obs:
            Observation tensor, shape ``[B, obs_dim]``.
        x_t:
            Noisy action sequence at diffusion step *t*, shape
            ``[B, horizon, act_dim]``.
        t:
            Diffusion timestep indices, shape ``[B]`` int64.

        Returns
        -------
        torch.Tensor
            Predicted noise eps_hat, shape ``[B, horizon, act_dim]``.
        """
        B = obs.shape[0]

        # Timestep conditioning.
        t_emb = sinusoidal_timestep_embedding(t, self.t_embed_dim)   # [B, t_embed_dim]
        t_cond = torch.relu(self.t_proj(t_emb))                      # [B, hidden_dim]

        # Flatten noisy sequence.
        x_flat = x_t.reshape(B, -1)                                  # [B, horizon*act_dim]

        # Concatenate all inputs.
        inp = torch.cat([obs, x_flat, t_cond], dim=1)                # [B, total_in]

        # Predict noise, reshape back to sequence form.
        out = self.mlp(inp)                                           # [B, horizon*act_dim]
        eps_hat = out.reshape(B, self.horizon, self.act_dim)
        return eps_hat
