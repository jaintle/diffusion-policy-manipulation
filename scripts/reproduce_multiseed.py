"""
Multi-seed experiment orchestrator.

Runs the full pipeline (record → train BC → eval BC → train diffusion →
eval diffusion) for each requested seed, writing results under:

    results/
        seed{N}/
            bc_eval.json
            diff_open_loop.json
            diff_receding.json

Usage
-----
    python scripts/reproduce_multiseed.py
    python scripts/reproduce_multiseed.py --seeds 0 1 2 --steps_bc 2000 --steps_diff 2000

Exit codes
----------
    0   All seeds completed successfully.
    1   At least one seed failed (error printed before exit).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS   = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command and exit on failure."""
    print(f"\n[reproduce] >>> {label}")
    print(f"[reproduce] CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[reproduce] FAILED ({label})  exit={result.returncode}")
        sys.exit(1)
    print(f"[reproduce] OK ({label})")


def _script(name: str) -> str:
    return os.path.join(_SCRIPTS, name)


# ---------------------------------------------------------------------------
# Per-seed pipeline
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    env_id: str,
    episodes_record: int,
    max_steps_record: int,
    steps_bc: int,
    steps_diff: int,
    episodes_eval: int,
    max_steps_eval: int,
    results_root: str,
    device: str,
) -> None:
    seed_dir    = os.path.join(results_root, f"seed{seed}")
    data_dir    = os.path.join(_REPO_ROOT, "data", "_repro")
    bc_run_dir  = os.path.join(_REPO_ROOT, "runs", "_repro", f"bc_seed{seed}")
    diff_run_dir = os.path.join(_REPO_ROOT, "runs", "_repro", f"diffusion_seed{seed}")
    dataset_path = os.path.join(data_dir, f"pusht_repro_seed{seed}.npz")

    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(bc_run_dir, exist_ok=True)
    os.makedirs(diff_run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Record dataset (skip if already present).
    # ------------------------------------------------------------------
    if os.path.exists(dataset_path):
        print(f"[reproduce] Seed {seed}: reusing dataset {dataset_path}")
    else:
        _run(
            [
                sys.executable, _script("record_dataset.py"),
                "--env_id",    env_id,
                "--seed",      str(seed),
                "--episodes",  str(episodes_record),
                "--max_steps", str(max_steps_record),
                "--out",       dataset_path,
            ],
            f"seed={seed} record_dataset",
        )

    # ------------------------------------------------------------------
    # 2. Train Gaussian BC.
    # ------------------------------------------------------------------
    bc_ckpt = os.path.join(bc_run_dir, "checkpoint.pt")
    if os.path.exists(bc_ckpt):
        print(f"[reproduce] Seed {seed}: reusing BC checkpoint {bc_ckpt}")
    else:
        _run(
            [
                sys.executable, _script("train_bc.py"),
                "--dataset_path", dataset_path,
                "--run_dir",      bc_run_dir,
                "--seed",         str(seed),
                "--steps",        str(steps_bc),
                "--device",       device,
            ],
            f"seed={seed} train_bc",
        )

    # ------------------------------------------------------------------
    # 3. Evaluate BC.
    # ------------------------------------------------------------------
    bc_eval_dest = os.path.join(seed_dir, "bc_eval.json")
    if os.path.exists(bc_eval_dest):
        print(f"[reproduce] Seed {seed}: reusing BC eval {bc_eval_dest}")
    else:
        bc_eval_tmp = os.path.join(bc_run_dir, "eval_results.json")
        _run(
            [
                sys.executable, _script("eval_bc.py"),
                "--env_id",          env_id,
                "--checkpoint_path", bc_ckpt,
                "--out_path",        bc_eval_tmp,
                "--seed",            str(seed),
                "--episodes",        str(episodes_eval),
                "--max_steps",       str(max_steps_eval),
                "--device",          device,
            ],
            f"seed={seed} eval_bc",
        )
        shutil.copy2(bc_eval_tmp, bc_eval_dest)
        print(f"[reproduce] Seed {seed}: BC eval → {bc_eval_dest}")

    # ------------------------------------------------------------------
    # 4. Train diffusion.
    # ------------------------------------------------------------------
    diff_ckpt = os.path.join(diff_run_dir, "checkpoint.pt")
    if os.path.exists(diff_ckpt):
        print(f"[reproduce] Seed {seed}: reusing diffusion checkpoint {diff_ckpt}")
    else:
        _run(
            [
                sys.executable, _script("train_diffusion.py"),
                "--dataset_path", dataset_path,
                "--run_dir",      diff_run_dir,
                "--seed",         str(seed),
                "--steps",        str(steps_diff),
                "--device",       device,
            ],
            f"seed={seed} train_diffusion",
        )

    # ------------------------------------------------------------------
    # 5. Evaluate diffusion (both execution modes).
    # ------------------------------------------------------------------
    diff_ol_dest = os.path.join(seed_dir, "diff_open_loop.json")
    diff_rh_dest = os.path.join(seed_dir, "diff_receding.json")

    if os.path.exists(diff_ol_dest) and os.path.exists(diff_rh_dest):
        print(f"[reproduce] Seed {seed}: reusing diffusion evals in {seed_dir}")
    else:
        diff_eval_tmp_dir = os.path.join(diff_run_dir, "eval_repro")
        _run(
            [
                sys.executable, _script("eval_diffusion.py"),
                "--env_id",           env_id,
                "--checkpoint_path",  diff_ckpt,
                "--out_dir",          diff_eval_tmp_dir,
                "--seed",             str(seed),
                "--episodes",         str(episodes_eval),
                "--max_steps",        str(max_steps_eval),
                "--device",           device,
            ],
            f"seed={seed} eval_diffusion",
        )
        # eval_diffusion writes eval_open_loop.json / eval_receding.json;
        # rename to the canonical names expected by validate/aggregate.
        shutil.copy2(
            os.path.join(diff_eval_tmp_dir, "eval_open_loop.json"),
            diff_ol_dest,
        )
        shutil.copy2(
            os.path.join(diff_eval_tmp_dir, "eval_receding.json"),
            diff_rh_dest,
        )
        print(f"[reproduce] Seed {seed}: diffusion evals → {seed_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment reproducer."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="List of integer seeds to run.",
    )
    parser.add_argument("--env_id",          type=str,   default="gym_pusht/PushT-v0")
    parser.add_argument("--episodes_record", type=int,   default=20)
    parser.add_argument("--max_steps_record",type=int,   default=200)
    parser.add_argument("--steps_bc",        type=int,   default=2000)
    parser.add_argument("--steps_diff",      type=int,   default=2000)
    parser.add_argument("--episodes_eval",   type=int,   default=10)
    parser.add_argument("--max_steps_eval",  type=int,   default=200)
    parser.add_argument(
        "--results_root", type=str, default="results",
        help="Root directory for per-seed output JSON files.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[reproduce] Seeds: {args.seeds}")
    print(f"[reproduce] Results root: {args.results_root}")

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"[reproduce] Starting seed {seed}")
        print(f"{'='*60}")
        run_seed(
            seed=seed,
            env_id=args.env_id,
            episodes_record=args.episodes_record,
            max_steps_record=args.max_steps_record,
            steps_bc=args.steps_bc,
            steps_diff=args.steps_diff,
            episodes_eval=args.episodes_eval,
            max_steps_eval=args.max_steps_eval,
            results_root=args.results_root,
            device=args.device,
        )

    print(f"\n[reproduce] All {len(args.seeds)} seed(s) completed successfully.")


if __name__ == "__main__":
    main()
