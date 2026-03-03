"""
Per-seed scatter plot of return_mean with mean marker overlay.

Reads:  results/rq_exec_mode/per_seed.csv
Writes: results/rq_exec_mode/per_seed_plot.png  (or --out_path)

Usage
-----
    python scripts/plot_per_seed.py
    python scripts/plot_per_seed.py \
        --per_seed_csv results/rq_exec_mode/per_seed.csv \
        --out_path     results/rq_exec_mode/per_seed_plot.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def load_per_seed(path: str) -> dict[str, list[float]]:
    """Return {method: [return_mean_per_seed, ...]} preserving row order."""
    data: dict[str, list[float]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = row["method"].strip()
            val_str = row.get("return_mean", "").strip()
            if val_str == "" or val_str.lower() == "nan":
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            data.setdefault(method, []).append(val)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-seed return_mean scatter plot.")
    parser.add_argument(
        "--per_seed_csv",
        default="results/rq_exec_mode/per_seed.csv",
    )
    parser.add_argument(
        "--out_path",
        default="results/rq_exec_mode/per_seed_plot.png",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.per_seed_csv):
        print(f"[plot_per_seed] ERROR: file not found: {args.per_seed_csv}")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_seed = load_per_seed(args.per_seed_csv)

    # ------------------------------------------------------------------
    # Method ordering and display labels.
    # ------------------------------------------------------------------
    order = [
        ("bc",             "BC"),
        ("diff_open_loop", "Diff Open-loop"),
        ("diff_receding",  "Diff Receding"),
    ]

    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    # Build filtered, ordered list of (x_position, label, values, color).
    plot_items: list[tuple[int, str, list[float], str]] = []
    for idx, (method_key, method_label) in enumerate(order):
        if method_key not in per_seed or not per_seed[method_key]:
            print(f"[plot_per_seed] WARNING: method '{method_key}' not found or empty; skipping.")
            continue
        plot_items.append((len(plot_items), method_label, per_seed[method_key], colors[idx]))

    if not plot_items:
        print("[plot_per_seed] ERROR: no plottable data found in per_seed CSV.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    x_positions: list[int] = []
    x_labels:    list[str] = []

    for x_pos, label, values, color in plot_items:
        x_positions.append(x_pos)
        x_labels.append(label)
        n = len(values)

        # Individual seed points — jittered horizontally for visibility.
        # Jitter is fully deterministic (evenly spaced around center).
        if n == 1:
            jitter = [0.0]
        else:
            half_spread = min(0.15, 0.08 * (n - 1))
            jitter = [
                -half_spread + i * (2 * half_spread / (n - 1))
                for i in range(n)
            ]

        for j_offset, val in zip(jitter, values):
            ax.scatter(
                x_pos + j_offset,
                val,
                color=color,
                s=40,
                alpha=0.75,
                zorder=3,
                linewidths=0.5,
                edgecolors="black",
            )

        # Mean marker.
        mean_val = sum(values) / len(values)
        ax.scatter(
            x_pos,
            mean_val,
            color=color,
            s=120,
            marker="D",
            zorder=4,
            linewidths=0.8,
            edgecolors="black",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Return per seed", fontsize=11)
    ax.set_title("Per-seed Returns (PushT)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Add a minimal legend distinguishing seed dots from the mean marker.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=7, label="seed"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="mean"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc="upper right")

    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fig.savefig(args.out_path, dpi=150)
    plt.close(fig)

    print(f"[plot_per_seed] saved → {args.out_path}")


if __name__ == "__main__":
    main()
