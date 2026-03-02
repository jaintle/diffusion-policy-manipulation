"""
CLI wrapper for BC training.

Usage
-----
    python scripts/train_bc.py
    python scripts/train_bc.py --dataset_path data/pusht_demo_seed0.npz \
        --run_dir runs/bc_run0 --seed 0 --steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.train.bc_trainer import train_bc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gaussian MLP BC policy.")
    parser.add_argument(
        "--dataset_path", type=str, default="data/_smoke/pusht_smoke_seed0.npz",
        help="Path to .npz dataset file.",
    )
    parser.add_argument(
        "--run_dir", type=str, default="runs/bc_smoke",
        help="Directory for checkpoint and logs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    checkpoint_path = train_bc(
        dataset_path=args.dataset_path,
        run_dir=args.run_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
