"""
Execution-strategy policy wrappers.

All wrappers implement the Policy protocol:
    reset(seed)  – called once per episode before the first act()
    act(obs)     – accepts/returns NumPy float32 arrays

Three wrappers are provided:

    GaussianBCPolicyWrapper
        Loads a Phase 3 checkpoint and always returns the deterministic mean
        action.  No internal RNG; reset() is a no-op.

    DiffusionOpenLoopPolicyWrapper
        Caches a full H-step action sequence from DDIM and executes all H
        actions before resampling.  Re-plans only when the cache is exhausted.
        Seed: episode_seed + chunk_index + sample_seed_base

    DiffusionRecedingHorizonPolicyWrapper
        Re-plans at every timestep; only the first action of each H-step
        sequence is executed.
        Seed: episode_seed + timestep_index + sample_seed_base
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from diffusion_policy_manipulation.models.diffusion_schedule import DiffusionSchedule
from diffusion_policy_manipulation.models.mlp_bc import GaussianMLPPolicy
from diffusion_policy_manipulation.models.samplers import sample_ddim


# ---------------------------------------------------------------------------
# Shared protocol
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Minimal policy interface used by rollout_evaluator."""

    def reset(self, seed: int) -> None:
        """Called once at the start of each episode."""
        ...

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return a flat action array given a flat observation array."""
        ...


# ---------------------------------------------------------------------------
# Gaussian MLP BC wrapper
# ---------------------------------------------------------------------------

class GaussianBCPolicyWrapper:
    """Wraps a trained GaussianMLPPolicy checkpoint for deterministic eval.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` file produced by Phase 3 ``train_bc``.
    device:
        Torch device string.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        dev = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)

        self._policy = GaussianMLPPolicy(
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        ).to(dev)
        self._policy.load_state_dict(ckpt["state_dict"])
        self._policy.eval()
        self._device = dev

    def reset(self, seed: int) -> None:
        """No internal state — no-op."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=self._device
        ).unsqueeze(0)  # [1, obs_dim]
        with torch.no_grad():
            action = self._policy.deterministic_action(obs_t)  # [1, act_dim]
        return action.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Diffusion open-loop chunk execution
# ---------------------------------------------------------------------------

class DiffusionOpenLoopPolicyWrapper:
    """Open-loop chunk execution: sample H actions, execute all, then resample.

    Parameters
    ----------
    denoiser:
        MLPDenoiser instance (trained or randomly initialized).
    schedule:
        Pre-computed DiffusionSchedule.
    horizon:
        Action sequence length H.
    act_dim:
        Dimensionality of a single action.
    ddim_steps:
        Number of DDIM denoising steps K.
    sample_seed_base:
        Fixed offset added to the per-chunk seed to avoid collisions with other
        wrappers.
    device:
        Torch device string.
    """

    def __init__(
        self,
        denoiser: torch.nn.Module,
        schedule: DiffusionSchedule,
        horizon: int,
        act_dim: int,
        ddim_steps: int,
        sample_seed_base: int,
        device: str = "cpu",
    ) -> None:
        self._denoiser = denoiser
        self._schedule = schedule
        self._horizon = horizon
        self._act_dim = act_dim
        self._ddim_steps = ddim_steps
        self._sample_seed_base = sample_seed_base
        self._device = torch.device(device)

        # Per-episode mutable state (reset each episode).
        self._episode_seed: int = 0
        self._action_cache: np.ndarray | None = None  # [horizon, act_dim]
        self._cache_ptr: int = 0
        self._chunk_index: int = 0

    def reset(self, seed: int) -> None:
        self._episode_seed = seed
        self._action_cache = None
        self._cache_ptr = 0
        self._chunk_index = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Resample when cache is empty or the current chunk has been consumed.
        if self._action_cache is None or self._cache_ptr >= self._horizon:
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=self._device
            ).unsqueeze(0)  # [1, obs_dim]

            sample_seed = (
                self._episode_seed + self._chunk_index + self._sample_seed_base
            )
            with torch.no_grad():
                seq = sample_ddim(
                    denoiser=self._denoiser,
                    schedule=self._schedule,
                    obs=obs_t,
                    horizon=self._horizon,
                    act_dim=self._act_dim,
                    steps=self._ddim_steps,
                    seed=sample_seed,
                )  # [1, horizon, act_dim]

            self._action_cache = seq.squeeze(0).cpu().numpy().astype(np.float32)
            self._cache_ptr = 0
            self._chunk_index += 1

        action = self._action_cache[self._cache_ptr]
        self._cache_ptr += 1
        return action  # [act_dim] float32


# ---------------------------------------------------------------------------
# Diffusion receding-horizon execution
# ---------------------------------------------------------------------------

class DiffusionRecedingHorizonPolicyWrapper:
    """Receding-horizon execution: resample at every step, execute first action.

    Parameters
    ----------
    denoiser:
        MLPDenoiser instance.
    schedule:
        Pre-computed DiffusionSchedule.
    horizon:
        Action sequence length H (only the first action is executed).
    act_dim:
        Dimensionality of a single action.
    ddim_steps:
        Number of DDIM denoising steps K.
    sample_seed_base:
        Fixed seed offset (distinct from the open-loop wrapper's base).
    device:
        Torch device string.
    """

    def __init__(
        self,
        denoiser: torch.nn.Module,
        schedule: DiffusionSchedule,
        horizon: int,
        act_dim: int,
        ddim_steps: int,
        sample_seed_base: int,
        device: str = "cpu",
    ) -> None:
        self._denoiser = denoiser
        self._schedule = schedule
        self._horizon = horizon
        self._act_dim = act_dim
        self._ddim_steps = ddim_steps
        self._sample_seed_base = sample_seed_base
        self._device = torch.device(device)

        self._episode_seed: int = 0
        self._timestep_index: int = 0

    def reset(self, seed: int) -> None:
        self._episode_seed = seed
        self._timestep_index = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(
            obs, dtype=torch.float32, device=self._device
        ).unsqueeze(0)  # [1, obs_dim]

        sample_seed = (
            self._episode_seed + self._timestep_index + self._sample_seed_base
        )
        with torch.no_grad():
            seq = sample_ddim(
                denoiser=self._denoiser,
                schedule=self._schedule,
                obs=obs_t,
                horizon=self._horizon,
                act_dim=self._act_dim,
                steps=self._ddim_steps,
                seed=sample_seed,
            )  # [1, horizon, act_dim]

        self._timestep_index += 1
        return seq[0, 0].cpu().numpy().astype(np.float32)  # first action [act_dim]
