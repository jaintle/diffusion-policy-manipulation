"""
CLI wrapper for diffusion policy training.

Usage
-----
    python scripts/train_diffusion.py
    python scripts/train_diffusion.py --dataset_path data/pusht_demo_seed0.npz \
        --run_dir runs/diffusion_run0 --steps 2000
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.train.diffusion_trainer import train_diffusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion MLP denoiser.")
    parser.add_argument(
        "--dataset_path", type=str, default="data/_smoke/pusht_smoke_seed0.npz",
    )
    parser.add_argument("--run_dir",     type=str,   default="runs/_smoke/diffusion_seed0")
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--steps",       type=int,   default=500)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--horizon",     type=int,   default=8)
    parser.add_argument("--t_embed_dim", type=int,   default=64)
    parser.add_argument("--T",           type=int,   default=50)
    parser.add_argument("--beta_start",  type=float, default=1e-4)
    parser.add_argument("--beta_end",    type=float, default=0.02)
    parser.add_argument("--device",      type=str,   default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    checkpoint_path = train_diffusion(
        dataset_path=args.dataset_path,
        run_dir=args.run_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        horizon=args.horizon,
        t_embed_dim=args.t_embed_dim,
        T=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=args.device,
    )
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
