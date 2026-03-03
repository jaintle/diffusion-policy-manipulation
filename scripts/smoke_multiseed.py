"""
Phase 7 Smoke Test – Multi-seed pipeline (2 seeds).
=====================================================

Verifies:
1. reproduce_multiseed.py completes for seeds [0, 1] without crash.
2. validate_results.py passes for seeds [0, 1].
3. aggregate_results.py writes per_seed.csv and summary.csv without crash.
4. Both CSV files are non-empty.

All artifacts are written under  results/_smoke_multiseed/  to avoid
polluting the main results/ tree.

Usage
-----
    python scripts/smoke_multiseed.py

Exit codes
----------
    0   All checks passed  ("Multi-seed smoke test PASSED")
    1   Any step failed    (details printed before exit)
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS   = os.path.dirname(os.path.abspath(__file__))

SMOKE_RESULTS = os.path.join(_REPO_ROOT, "results", "_smoke_multiseed")
SEEDS = [0, 1]


def _fail(msg: str) -> None:
    print(f"[smoke_multiseed] FAIL  {msg}")
    sys.exit(1)


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[smoke_multiseed] >>> {label}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        _fail(f"{label}  exit={result.returncode}")
    print(f"[smoke_multiseed] OK  {label}")


def _script(name: str) -> str:
    return os.path.join(_SCRIPTS, name)


def main() -> None:
    os.makedirs(SMOKE_RESULTS, exist_ok=True)

    seeds_str = [str(s) for s in SEEDS]

    # ------------------------------------------------------------------
    # 1.  reproduce_multiseed  (tiny training run: 100 steps each)
    # ------------------------------------------------------------------
    _run(
        [
            sys.executable, _script("reproduce_multiseed.py"),
            "--seeds",            *seeds_str,
            "--env_id",           "gym_pusht/PushT-v0",
            "--episodes_record",  "3",
            "--max_steps_record", "50",
            "--steps_bc",         "100",
            "--steps_diff",       "100",
            "--episodes_eval",    "3",
            "--max_steps_eval",   "50",
            "--results_root",     SMOKE_RESULTS,
            "--device",           "cpu",
        ],
        "reproduce_multiseed",
    )

    # ------------------------------------------------------------------
    # 2.  validate_results
    # ------------------------------------------------------------------
    _run(
        [
            sys.executable, _script("validate_results.py"),
            "--seeds",        *seeds_str,
            "--results_root", SMOKE_RESULTS,
        ],
        "validate_results",
    )

    # ------------------------------------------------------------------
    # 3.  aggregate_results
    # ------------------------------------------------------------------
    _run(
        [
            sys.executable, _script("aggregate_results.py"),
            "--seeds",        *seeds_str,
            "--results_root", SMOKE_RESULTS,
        ],
        "aggregate_results",
    )

    # ------------------------------------------------------------------
    # 4.  Verify CSV outputs exist and are non-empty.
    # ------------------------------------------------------------------
    for csv_name in ("per_seed.csv", "summary.csv"):
        csv_path = os.path.join(SMOKE_RESULTS, csv_name)
        if not os.path.isfile(csv_path):
            _fail(f"Expected CSV not found: {csv_path}")
        with open(csv_path, newline="") as fh:
            rows = list(csv.reader(fh))
        # Expect at least 1 header row + 1 data row.
        if len(rows) < 2:
            _fail(f"{csv_name} is empty (only header or no rows): {csv_path}")
        print(f"[smoke_multiseed] {csv_name}  rows={len(rows)-1} (excl. header)  OK")

    print("\n[smoke_multiseed] Multi-seed smoke test PASSED")


if __name__ == "__main__":
    main()
