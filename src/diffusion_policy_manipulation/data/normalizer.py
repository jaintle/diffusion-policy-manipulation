"""
Observation and action normalizer.

RunningNormalizer computes mean/std statistics from a dataset and supports
saving/loading those statistics to/from JSON so normalization is reproducible
across runs without re-fitting.

NumPy + json only — no torch dependency.
"""

from __future__ import annotations

import json
import os

import numpy as np


class RunningNormalizer:
    """Fit mean/std statistics for obs and actions, then normalize on demand.

    Usage
    -----
    ::

        norm = RunningNormalizer()
        norm.fit(obs, actions)          # compute statistics
        norm_obs = norm.normalize_obs(obs)
        norm.save("stats/normalizer.json")

        norm2 = RunningNormalizer.load("stats/normalizer.json")
        norm_obs2 = norm2.normalize_obs(obs)   # identical to norm_obs
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

        # Populated by fit() or load().
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, obs: np.ndarray, actions: np.ndarray) -> None:
        """Compute mean and std along axis 0 for obs and actions.

        Parameters
        ----------
        obs:
            Array of shape [N, obs_dim].
        actions:
            Array of shape [N, act_dim].
        """
        self.obs_mean = obs.mean(axis=0)
        self.obs_std = np.maximum(obs.std(axis=0), self.eps)

        self.act_mean = actions.mean(axis=0)
        self.act_std = np.maximum(actions.std(axis=0), self.eps)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Standardize observations: ``(obs - mean) / std``."""
        self._check_fitted()
        return (obs - self.obs_mean) / self.obs_std

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Standardize actions: ``(actions - mean) / std``."""
        self._check_fitted()
        return (actions - self.act_mean) / self.act_std

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize statistics to a JSON file.

        Keys: ``obs_mean``, ``obs_std``, ``act_mean``, ``act_std``.
        Values are plain Python lists (float64 precision).
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "RunningNormalizer":
        """Deserialize statistics from a JSON file produced by :meth:`save`.

        Returns a fitted ``RunningNormalizer`` instance.
        """
        with open(path) as fh:
            payload = json.load(fh)

        norm = cls()
        norm.obs_mean = np.array(payload["obs_mean"], dtype=np.float64)
        norm.obs_std = np.array(payload["obs_std"], dtype=np.float64)
        norm.act_mean = np.array(payload["act_mean"], dtype=np.float64)
        norm.act_std = np.array(payload["act_std"], dtype=np.float64)
        return norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.obs_mean is None:
            raise RuntimeError(
                "RunningNormalizer has not been fitted. Call fit() or load() first."
            )
