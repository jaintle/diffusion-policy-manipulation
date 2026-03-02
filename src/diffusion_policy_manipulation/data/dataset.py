"""
Minimal dataset loader for .npz trajectory files produced by record_dataset.py.

NpzTrajectoryDataset:
    - Loads and validates the .npz format.
    - Provides deterministic batch sampling via a seed-controlled RandomState.
"""

from __future__ import annotations

import numpy as np


class NpzTrajectoryDataset:
    """Read-only dataset backed by a single .npz transition file.

    Expected .npz keys
    ------------------
    obs         float32  [N, obs_dim]
    actions     float32  [N, act_dim]
    rewards     float32  [N]
    terminated  uint8    [N]
    truncated   uint8    [N]
    next_obs    float32  [N, obs_dim]
    episode_id  int32    [N]
    timestep    int32    [N]
    """

    REQUIRED_KEYS = (
        "obs",
        "actions",
        "rewards",
        "terminated",
        "truncated",
        "next_obs",
        "episode_id",
        "timestep",
    )

    def __init__(self, path: str) -> None:
        data = np.load(path)

        # Validate all required keys are present.
        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"NpzTrajectoryDataset: missing keys {missing} in '{path}'"
            )

        # Load and cast to canonical dtypes.
        self._obs = data["obs"].astype(np.float32)
        self._actions = data["actions"].astype(np.float32)
        self._rewards = data["rewards"].astype(np.float32)
        self._terminated = data["terminated"]
        self._truncated = data["truncated"]
        self._next_obs = data["next_obs"].astype(np.float32)
        self._episode_id = data["episode_id"]
        self._timestep = data["timestep"]

        # Validate all arrays share the same leading dimension.
        N = len(self._obs)
        for key in self.REQUIRED_KEYS:
            arr_len = len(data[key])
            if arr_len != N:
                raise ValueError(
                    f"NpzTrajectoryDataset: key '{key}' has length {arr_len}, "
                    f"expected {N}"
                )

        self._N = N

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._N

    def sample_batch(self, batch_size: int, seed: int) -> dict[str, np.ndarray]:
        """Sample a batch of transitions deterministically.

        Parameters
        ----------
        batch_size:
            Number of transitions to return.
        seed:
            Seed for the local RandomState used to draw indices.  The same
            seed always produces the same batch.

        Returns
        -------
        dict with keys: obs, actions, rewards, terminated, truncated, next_obs.
        All float arrays are float32; flag arrays keep their original dtype.
        """
        rng = np.random.RandomState(seed)
        indices = rng.choice(self._N, size=batch_size, replace=True)

        return {
            "obs": self._obs[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "terminated": self._terminated[indices],
            "truncated": self._truncated[indices],
            "next_obs": self._next_obs[indices],
        }
