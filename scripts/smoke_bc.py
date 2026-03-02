"""
Phase 3 Smoke Test – BC Training and Evaluation Determinism
============================================================

Verifies:
1. A small dataset can be recorded (or reused from Phase 2).
2. BC training completes without crash and writes required artefacts.
3. Two independent evaluation runs with the same checkpoint + seed produce
   exactly identical JSON outputs.

Usage
-----
    python scripts/smoke_bc.py
    python scripts/smoke_bc.py --env_id gym_pusht/PushT-v0 --seed 0

Exit codes
----------
    0   All checks passed  ("BC smoke test PASSED")
    1   A mismatch or crash was detected (details printed before exit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make src and scripts importable from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_manipulation.eval.bc_evaluator import evaluate_bc   # noqa: E402
from diffusion_policy_manipulation.train.bc_trainer import train_bc        # noqa: E402
from record_dataset import record                                          # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 BC smoke test.")
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master seed (default: 0).",
    )
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _compare_json(path_a: str, path_b: str) -> None:
    """Assert that two eval JSON files are field-for-field identical."""
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)

    for key in a:
        if key not in b:
            _fail(f"Eval JSON key '{key}' present in eval_a but missing in eval_b.")
        val_a, val_b = a[key], b[key]
        if isinstance(val_a, list):
            if val_a != val_b:
                _fail(
                    f"Eval JSON list mismatch: key='{key}'\n"
                    f"  eval_a: {val_a}\n"
                    f"  eval_b: {val_b}"
                )
        elif isinstance(val_a, float):
            if val_a != val_b:
                _fail(
                    f"Eval JSON float mismatch: key='{key}' "
                    f"eval_a={val_a} eval_b={val_b}"
                )
        else:
            if val_a != val_b:
                _fail(
                    f"Eval JSON value mismatch: key='{key}' "
                    f"eval_a={val_a!r} eval_b={val_b!r}"
                )

    # Also check for keys only in b.
    for key in b:
        if key not in a:
            _fail(f"Eval JSON key '{key}' present in eval_b but missing in eval_a.")


def main() -> None:
    args = parse_args()
    env_id = args.env_id
    seed = args.seed

    # ------------------------------------------------------------------
    # A) Ensure dataset exists (reuse Phase 2 file if present).
    # ------------------------------------------------------------------
    dataset_path = f"data/_smoke/pusht_smoke_seed{seed}.npz"
    os.makedirs("data/_smoke", exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Dataset not found — recording to {dataset_path} ...")
        record(
            env_id=env_id,
            seed=seed,
            episodes=3,
            max_steps=50,
            out=dataset_path,
        )
    else:
        print(f"Reusing existing dataset: {dataset_path}")

    # ------------------------------------------------------------------
    # B) Train BC briefly.
    # ------------------------------------------------------------------
    run_dir = f"runs/_smoke/bc_seed{seed}"
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = train_bc(
        dataset_path=dataset_path,
        run_dir=run_dir,
        seed=seed,
        batch_size=64,
        steps=200,
        lr=3e-4,
        hidden_dim=256,
        num_layers=2,
        device="cpu",
    )

    # Verify expected artefacts are present.
    for fname in ("checkpoint.pt", "config.json", "train_summary.json"):
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath):
            _fail(f"Expected training artefact not found: {fpath}")

    # ------------------------------------------------------------------
    # C) Evaluate twice with identical settings.
    # ------------------------------------------------------------------
    eval_a = os.path.join(run_dir, "eval_a.json")
    eval_b = os.path.join(run_dir, "eval_b.json")

    eval_kwargs = dict(
        env_id=env_id,
        checkpoint_path=checkpoint_path,
        seed=seed,
        episodes=5,
        max_steps=50,
        device="cpu",
    )

    evaluate_bc(**eval_kwargs, out_path=eval_a)
    evaluate_bc(**eval_kwargs, out_path=eval_b)

    # ------------------------------------------------------------------
    # D) Assert outputs are identical.
    # ------------------------------------------------------------------
    _compare_json(eval_a, eval_b)

    print("BC smoke test PASSED")


if __name__ == "__main__":
    main()
