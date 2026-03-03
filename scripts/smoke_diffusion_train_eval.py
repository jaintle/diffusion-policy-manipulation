"""
Phase 6 Smoke Test – Diffusion Training and Evaluation Determinism
==================================================================

Verifies:
1. Diffusion training completes without crash and writes required artefacts.
2. Evaluation under both execution modes runs without crash.
3. Two independent evaluation runs with the same checkpoint + seed produce
   exactly identical JSON outputs for BOTH execution modes.

Usage
-----
    python scripts/smoke_diffusion_train_eval.py
    python scripts/smoke_diffusion_train_eval.py --env_id gym_pusht/PushT-v0 --seed 0

Exit codes
----------
    0   All checks passed  ("Diffusion train+eval smoke test PASSED")
    1   A mismatch or crash was detected (details printed before exit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_manipulation.train.diffusion_trainer import train_diffusion  # noqa: E402
from eval_diffusion import run_diffusion_eval                                      # noqa: E402
from record_dataset import record                                                  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 diffusion smoke test.")
    parser.add_argument("--env_id", type=str, default="gym_pusht/PushT-v0")
    parser.add_argument("--seed",   type=int, default=0)
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _compare_json_files(path_a: str, path_b: str, label: str) -> None:
    """Load two JSON files and assert all fields are exactly equal."""
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)

    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        if key not in a:
            _fail(f"{label} [{key}]: present in run_b but missing in run_a.")
        if key not in b:
            _fail(f"{label} [{key}]: present in run_a but missing in run_b.")
        va, vb = a[key], b[key]
        if isinstance(va, list):
            if va != vb:
                _fail(
                    f"{label} [{key}]: list mismatch.\n"
                    f"  run_a: {va}\n  run_b: {vb}"
                )
        elif isinstance(va, float):
            if va != vb:
                _fail(f"{label} [{key}]: float mismatch  run_a={va}  run_b={vb}")
        else:
            if va != vb:
                _fail(f"{label} [{key}]: value mismatch  run_a={va!r}  run_b={vb!r}")


def main() -> None:
    args = parse_args()
    env_id = args.env_id
    seed   = args.seed

    # ------------------------------------------------------------------
    # A) Ensure deterministic dataset exists.
    # ------------------------------------------------------------------
    dataset_path = f"data/_smoke/pusht_smoke_seed{seed}.npz"
    os.makedirs("data/_smoke", exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Dataset not found — recording to {dataset_path} ...")
        record(env_id=env_id, seed=seed, episodes=3, max_steps=50, out=dataset_path)
    else:
        print(f"Reusing existing dataset: {dataset_path}")

    # ------------------------------------------------------------------
    # B) Train diffusion briefly.
    # ------------------------------------------------------------------
    run_dir = f"runs/_smoke/diffusion_seed{seed}"
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = train_diffusion(
        dataset_path=dataset_path,
        run_dir=run_dir,
        seed=seed,
        batch_size=64,
        steps=300,
        lr=3e-4,
        hidden_dim=128,
        num_layers=2,
        horizon=8,
        t_embed_dim=64,
        T=50,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    )

    # Verify expected artefacts.
    for fname in ("checkpoint.pt", "config.json", "train_summary.json"):
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath):
            _fail(f"Expected training artefact not found: {fpath}")

    # ------------------------------------------------------------------
    # C) Evaluate twice with identical settings into separate directories.
    # ------------------------------------------------------------------
    eval_kwargs = dict(
        env_id=env_id,
        checkpoint_path=checkpoint_path,
        seed=seed,
        episodes=3,
        max_steps=50,
        K=10,
        sample_seed_base=999,
        device="cpu",
    )

    out_dir_a = os.path.join(run_dir, "eval_run_a")
    out_dir_b = os.path.join(run_dir, "eval_run_b")

    print("\n--- Eval run A ---")
    run_diffusion_eval(out_dir=out_dir_a, **eval_kwargs)

    print("\n--- Eval run B ---")
    run_diffusion_eval(out_dir=out_dir_b, **eval_kwargs)

    # ------------------------------------------------------------------
    # D) Assert exact equality across both runs.
    # ------------------------------------------------------------------
    for fname, label in [
        ("eval_open_loop.json", "open-loop"),
        ("eval_receding.json",  "receding-horizon"),
    ]:
        _compare_json_files(
            os.path.join(out_dir_a, fname),
            os.path.join(out_dir_b, fname),
            label,
        )

    print("\nDiffusion train+eval smoke test PASSED")


if __name__ == "__main__":
    main()
