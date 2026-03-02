"""
Phase 2 Smoke Test – Dataset Determinism Verification
======================================================

Verifies three invariants:

1. Dataset file determinism:
   Same seed  →  byte-identical arrays across two independently recorded files.

2. Batch sampling determinism:
   Same (dataset, seed) pair  →  identical sampled batches.

3. Normalizer round-trip:
   save() then load() reproduces statistics exactly.

Usage
-----
    python scripts/smoke_dataset.py
    python scripts/smoke_dataset.py --env_id gym_pusht/PushT-v0 --seed 0 \
        --episodes 3 --max_steps 50 --batch_size 32

Exit codes
----------
    0   All checks passed  ("Dataset smoke test PASSED")
    1   A mismatch was detected (details printed before exit)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make src and scripts importable when run from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_manipulation.data.dataset import NpzTrajectoryDataset    # noqa: E402
from diffusion_policy_manipulation.data.normalizer import RunningNormalizer     # noqa: E402
from record_dataset import record                                               # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dataset determinism for Phase 2."
    )
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master seed (default: 0).",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Episodes to record per run (default: 3).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=50,
        help="Max steps per episode (default: 50).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for sampling determinism check (default: 32).",
    )
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    env_id = args.env_id
    seed = args.seed
    episodes = args.episodes
    max_steps = args.max_steps
    batch_size = args.batch_size

    # ------------------------------------------------------------------
    # A) Prepare smoke output directory.
    # ------------------------------------------------------------------
    smoke_dir = "data/_smoke"
    os.makedirs(smoke_dir, exist_ok=True)

    file_a = os.path.join(smoke_dir, f"pusht_smoke_seed{seed}_a.npz")
    file_b = os.path.join(smoke_dir, f"pusht_smoke_seed{seed}_b.npz")
    norm_path = os.path.join(smoke_dir, f"normalizer_seed{seed}.json")

    # ------------------------------------------------------------------
    # B) Record twice with identical arguments.
    # ------------------------------------------------------------------
    record(env_id=env_id, seed=seed, episodes=episodes, max_steps=max_steps, out=file_a)
    record(env_id=env_id, seed=seed, episodes=episodes, max_steps=max_steps, out=file_b)

    # ------------------------------------------------------------------
    # C) Load both datasets.
    # ------------------------------------------------------------------
    ds_a = NpzTrajectoryDataset(file_a)
    ds_b = NpzTrajectoryDataset(file_b)

    # ------------------------------------------------------------------
    # D-1) Dataset file determinism.
    # ------------------------------------------------------------------
    raw_a = np.load(file_a)
    raw_b = np.load(file_b)

    float_keys = ("obs", "actions", "rewards", "next_obs")
    int_keys = ("terminated", "truncated", "episode_id", "timestep")

    for key in float_keys:
        a, b = raw_a[key], raw_b[key]
        if not np.allclose(a, b, atol=0.0, rtol=0.0):
            diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            _fail(f"Dataset file mismatch: key='{key}'  max|a-b|={diff:.6e}")

    for key in int_keys:
        if not np.array_equal(raw_a[key], raw_b[key]):
            _fail(f"Dataset file mismatch: key='{key}'")

    # ------------------------------------------------------------------
    # D-2) Batch sampling determinism.
    # ------------------------------------------------------------------
    batch_seed = seed + 999
    batch1 = ds_a.sample_batch(batch_size, seed=batch_seed)
    batch2 = ds_a.sample_batch(batch_size, seed=batch_seed)

    for key, arr1 in batch1.items():
        arr2 = batch2[key]
        if arr1.dtype.kind == "f":
            if not np.allclose(arr1, arr2, atol=0.0, rtol=0.0):
                _fail(f"Batch sampling mismatch: key='{key}'")
        else:
            if not np.array_equal(arr1, arr2):
                _fail(f"Batch sampling mismatch: key='{key}'")

    # ------------------------------------------------------------------
    # E) Normalizer round-trip.
    # ------------------------------------------------------------------
    norm = RunningNormalizer()
    norm.fit(ds_a._obs, ds_a._actions)
    norm.save(norm_path)

    norm2 = RunningNormalizer.load(norm_path)

    for attr in ("obs_mean", "obs_std", "act_mean", "act_std"):
        v1 = getattr(norm, attr)
        v2 = getattr(norm2, attr)
        if not np.allclose(v1, v2, atol=0.0, rtol=0.0):
            _fail(f"Normalizer round-trip mismatch: '{attr}'")

    print("Dataset smoke test PASSED")


if __name__ == "__main__":
    main()
