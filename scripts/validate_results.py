"""
Validate that all required result files exist for each requested seed.

Expected layout
---------------
    results/
        seed{N}/
            bc_eval.json
            diff_open_loop.json
            diff_receding.json

Usage
-----
    python scripts/validate_results.py
    python scripts/validate_results.py --seeds 0 1 2 --results_root results

Exit codes
----------
    0   All files present and valid JSON.
    1   One or more files missing or unreadable (details printed before exit).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


REQUIRED_FILES = [
    "bc_eval.json",
    "diff_open_loop.json",
    "diff_receding.json",
]

REQUIRED_KEYS = {
    "bc_eval.json":         {"success_rate", "return_mean", "return_std", "episode_len_mean"},
    "diff_open_loop.json":  {"success_rate", "return_mean", "return_std", "episode_len_mean"},
    "diff_receding.json":   {"success_rate", "return_mean", "return_std", "episode_len_mean"},
}


def _fail(msg: str) -> None:
    print(f"[validate] FAIL  {msg}")
    sys.exit(1)


def validate_seed(seed: int, results_root: str) -> None:
    seed_dir = os.path.join(results_root, f"seed{seed}")

    if not os.path.isdir(seed_dir):
        _fail(f"seed={seed}: directory not found: {seed_dir}")

    for fname in REQUIRED_FILES:
        fpath = os.path.join(seed_dir, fname)

        # --- existence ---
        if not os.path.isfile(fpath):
            _fail(f"seed={seed}: missing file: {fpath}")

        # --- parseable JSON ---
        try:
            with open(fpath) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            _fail(f"seed={seed}: invalid JSON in {fpath}: {exc}")

        # --- required keys ---
        for key in REQUIRED_KEYS.get(fname, set()):
            if key not in data:
                _fail(
                    f"seed={seed}: key '{key}' missing in {fpath}\n"
                    f"  present keys: {sorted(data.keys())}"
                )

        print(f"[validate] OK  seed={seed}  {fname}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-seed result files."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
    )
    parser.add_argument(
        "--results_root", type=str, default="results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        validate_seed(seed, args.results_root)
    print(f"\n[validate] All {len(args.seeds)} seed(s) passed validation.")


if __name__ == "__main__":
    main()
