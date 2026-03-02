"""
CLI wrapper for deterministic BC evaluation.

Usage
-----
    python scripts/eval_bc.py --checkpoint_path runs/bc_smoke/checkpoint.pt
    python scripts/eval_bc.py --checkpoint_path runs/bc_smoke/checkpoint.pt \
        --env_id gym_pusht/PushT-v0 --seed 0 --episodes 10 --max_steps 200
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.eval.bc_evaluator import evaluate_bc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BC checkpoint.")
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to checkpoint.pt produced by train_bc.",
    )
    parser.add_argument(
        "--out_path", type=str, default="runs/bc_smoke/eval.json",
        help="Path for the output JSON results file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    evaluate_bc(
        env_id=args.env_id,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_path=args.out_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
