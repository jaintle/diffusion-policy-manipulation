"""
Aggregate per-seed results into summary statistics.

Reads
-----
    results/seed{N}/bc_eval.json
    results/seed{N}/diff_open_loop.json
    results/seed{N}/diff_receding.json

Writes
------
    results/per_seed.csv         — raw values for every seed × method
    results/summary.csv          — mean ± std across seeds for every method

Aggregation rules
-----------------
- If ANY seed directory or required file is missing, aggregation FAILS.
- Numeric fields aggregated: success_rate, return_mean, return_std,
  episode_len_mean, mean_policy_time (optional).
- summary.csv contains: method, metric, mean, std, n_seeds.

Usage
-----
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --seeds 0 1 2 --results_root results

Exit codes
----------
    0   Outputs written successfully.
    1   Missing file or field (details printed before exit).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


METHODS: list[tuple[str, str]] = [
    ("bc",           "bc_eval.json"),
    ("diff_open_loop", "diff_open_loop.json"),
    ("diff_receding",  "diff_receding.json"),
]

NUMERIC_FIELDS = [
    "success_rate",
    "return_mean",
    "return_std",
    "episode_len_mean",
    "mean_policy_time",   # optional — skip if absent
]

REQUIRED_FIELDS = {"success_rate", "return_mean", "return_std", "episode_len_mean"}


def _fail(msg: str) -> None:
    print(f"[aggregate] FAIL  {msg}")
    sys.exit(1)


def load_seed_method(seed: int, fname: str, results_root: str) -> dict:
    fpath = os.path.join(results_root, f"seed{seed}", fname)
    if not os.path.isfile(fpath):
        _fail(f"seed={seed}: missing file: {fpath}")
    try:
        with open(fpath) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        _fail(f"seed={seed}: invalid JSON in {fpath}: {exc}")
    for key in REQUIRED_FIELDS:
        if key not in data:
            _fail(f"seed={seed}: required key '{key}' missing in {fpath}")
    return data


def main() -> None:
    args = parse_args()
    seeds = args.seeds
    results_root = args.results_root

    # ------------------------------------------------------------------ load
    # raw[method_name][field] = [val_seed0, val_seed1, ...]
    raw: dict[str, dict[str, list[float]]] = {}

    for method_name, fname in METHODS:
        raw[method_name] = {f: [] for f in NUMERIC_FIELDS}
        for seed in seeds:
            data = load_seed_method(seed, fname, results_root)
            for field in NUMERIC_FIELDS:
                if field in data:
                    raw[method_name][field].append(float(data[field]))
                else:
                    raw[method_name][field].append(float("nan"))

    # --------------------------------------------------------------- per_seed
    per_seed_path = os.path.join(results_root, "per_seed.csv")
    per_seed_rows: list[dict] = []

    for method_name, fname in METHODS:
        for i, seed in enumerate(seeds):
            row: dict = {"method": method_name, "seed": seed}
            for field in NUMERIC_FIELDS:
                val = raw[method_name][field][i]
                row[field] = "" if (val != val) else val   # NaN → blank
            per_seed_rows.append(row)

    _write_csv(
        per_seed_path,
        fieldnames=["method", "seed"] + NUMERIC_FIELDS,
        rows=per_seed_rows,
    )
    print(f"[aggregate] per_seed.csv → {per_seed_path}")

    # ---------------------------------------------------------------- summary
    summary_path = os.path.join(results_root, "summary.csv")
    summary_rows: list[dict] = []

    for method_name, _ in METHODS:
        for field in NUMERIC_FIELDS:
            vals = [v for v in raw[method_name][field] if v == v]  # drop NaN
            if not vals:
                continue   # field entirely absent across all seeds — skip row
            arr  = np.array(vals, dtype=float)
            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=0))
            summary_rows.append({
                "method":  method_name,
                "metric":  field,
                "mean":    mean,
                "std":     std,
                "n_seeds": len(vals),
            })

    _write_csv(
        summary_path,
        fieldnames=["method", "metric", "mean", "std", "n_seeds"],
        rows=summary_rows,
    )
    print(f"[aggregate] summary.csv  → {summary_path}")

    # ----------------------------------------------------------------- print
    _print_summary(summary_rows)

    print(f"\n[aggregate] Done.  ({len(seeds)} seed(s))")


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary_rows: list[dict]) -> None:
    print("\n--- Aggregated Results ---")
    current_method = None
    for row in summary_rows:
        if row["method"] != current_method:
            current_method = row["method"]
            print(f"\n  {current_method}")
        print(
            f"    {row['metric']:22s}  "
            f"mean={row['mean']:.4f}  "
            f"std={row['std']:.4f}  "
            f"(n={row['n_seeds']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed JSON results into CSV summaries."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
    )
    parser.add_argument(
        "--results_root", type=str, default="results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
