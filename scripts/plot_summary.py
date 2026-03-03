"""
Bar plot of return_mean (mean ± std over seeds) for each method.

Reads:  results/rq_exec_mode/summary.csv
Writes: results/rq_exec_mode/summary_plot.png  (or --out_path)

Usage
-----
    python scripts/plot_summary.py
    python scripts/plot_summary.py \
        --summary_csv results/rq_exec_mode/summary.csv \
        --out_path    results/rq_exec_mode/summary_plot.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def load_summary(path: str) -> dict[str, dict[str, float]]:
    """Return {method: {metric: value}} for mean and std columns."""
    data: dict[str, dict[str, float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = row["method"].strip()
            metric = row["metric"].strip()
            if method not in data:
                data[method] = {}
            data[method][metric + "_mean"] = float(row["mean"])
            data[method][metric + "_std"]  = float(row["std"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar plot of aggregated return_mean.")
    parser.add_argument(
        "--summary_csv",
        default="results/rq_exec_mode/summary.csv",
    )
    parser.add_argument(
        "--out_path",
        default="results/rq_exec_mode/summary_plot.png",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.summary_csv):
        print(f"[plot_summary] ERROR: file not found: {args.summary_csv}")
        sys.exit(1)

    # Import here so the script fails fast on missing file before loading heavy deps.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_summary(args.summary_csv)

    # ------------------------------------------------------------------
    # Method ordering and display labels.
    # ------------------------------------------------------------------
    order = [
        ("bc",             "BC"),
        ("diff_open_loop", "Diff Open-loop"),
        ("diff_receding",  "Diff Receding"),
    ]

    means: list[float] = []
    stds:  list[float] = []
    labels: list[str]  = []

    for method_key, method_label in order:
        if method_key not in data:
            print(f"[plot_summary] WARNING: method '{method_key}' not found in CSV; skipping.")
            continue
        method_data = data[method_key]
        if "return_mean_mean" not in method_data:
            print(f"[plot_summary] WARNING: 'return_mean' metric missing for '{method_key}'; skipping.")
            continue
        means.append(method_data["return_mean_mean"])
        stds.append(method_data.get("return_mean_std", 0.0))
        labels.append(method_label)

    if not means:
        print("[plot_summary] ERROR: no plottable data found in summary CSV.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        color=["#4878CF", "#6ACC65", "#D65F5F"][: len(labels)],
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
        width=0.55,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Return (mean \u00b1 std over seeds)", fontsize=11)
    ax.set_title("Execution Strategy Comparison (PushT)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fig.savefig(args.out_path, dpi=150)
    plt.close(fig)

    print(f"[plot_summary] saved → {args.out_path}")


if __name__ == "__main__":
    main()
