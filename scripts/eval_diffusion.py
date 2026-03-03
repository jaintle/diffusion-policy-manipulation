"""
CLI wrapper for deterministic diffusion policy evaluation.

Loads a diffusion checkpoint, reconstructs the schedule and denoiser, wraps
them with both execution modes (open-loop and receding-horizon), runs
evaluate_policy for each, and writes JSON outputs.

Usage
-----
    python scripts/eval_diffusion.py \\
        --checkpoint_path runs/_smoke/diffusion_seed0/checkpoint.pt

    python scripts/eval_diffusion.py \\
        --checkpoint_path runs/diffusion_run0/checkpoint.pt \\
        --out_dir runs/diffusion_run0 --episodes 10 --max_steps 200

Outputs
-------
    {out_dir}/eval_open_loop.json
    {out_dir}/eval_receding.json

Determinism guarantee
---------------------
Running with the same arguments twice produces byte-identical JSON files
because:
  - denoiser weights are fixed (loaded from checkpoint)
  - per-episode reset seeds are deterministic (seed + episode_index)
  - DDIM sampling seeds are deterministic (controlled by wrapper seed arithmetic)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.eval.policy_wrappers import (            # noqa: E402
    DiffusionOpenLoopPolicyWrapper,
    DiffusionRecedingHorizonPolicyWrapper,
)
from diffusion_policy_manipulation.eval.rollout_evaluator import evaluate_policy  # noqa: E402
from diffusion_policy_manipulation.models.diffusion_denoiser import MLPDenoiser   # noqa: E402
from diffusion_policy_manipulation.models.diffusion_schedule import (             # noqa: E402
    make_linear_schedule,
)


def run_diffusion_eval(
    env_id: str,
    checkpoint_path: str,
    out_dir: str,
    seed: int,
    episodes: int,
    max_steps: int,
    K: int,
    sample_seed_base: int,
    device: str,
) -> tuple[dict, dict]:
    """Load a checkpoint and evaluate both execution modes.

    Parameters
    ----------
    env_id:
        Registered Gymnasium environment ID.
    checkpoint_path:
        Path to the ``.pt`` file produced by ``train_diffusion``.
    out_dir:
        Directory where JSON results are written.
    seed:
        Master evaluation seed.
    episodes:
        Number of episodes per policy.
    max_steps:
        Maximum steps per episode.
    K:
        Number of DDIM denoising steps.
    sample_seed_base:
        Fixed offset for diffusion sample seeds.
    device:
        Torch device string.

    Returns
    -------
    tuple[dict, dict]
        (open_loop_results, receding_results) dicts.
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------- load checkpoint
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)

    schedule = make_linear_schedule(
        T=ckpt["T"],
        beta_start=ckpt["beta_start"],
        beta_end=ckpt["beta_end"],
        device=device,
    )

    denoiser = MLPDenoiser(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        horizon=ckpt["horizon"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        t_embed_dim=ckpt["t_embed_dim"],
    ).to(dev)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()

    horizon  = ckpt["horizon"]
    act_dim  = ckpt["act_dim"]

    # -------------------------------------------- instantiate policy wrappers
    open_loop_policy = DiffusionOpenLoopPolicyWrapper(
        denoiser=denoiser,
        schedule=schedule,
        horizon=horizon,
        act_dim=act_dim,
        ddim_steps=K,
        sample_seed_base=sample_seed_base,
        device=device,
    )

    receding_policy = DiffusionRecedingHorizonPolicyWrapper(
        denoiser=denoiser,
        schedule=schedule,
        horizon=horizon,
        act_dim=act_dim,
        ddim_steps=K,
        sample_seed_base=sample_seed_base + 100_000,
        device=device,
    )

    # ------------------------------------------------------ run evaluations
    eval_kwargs = dict(
        env_id=env_id,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
    )

    print("Evaluating open-loop policy ...")
    ol_results = evaluate_policy(policy=open_loop_policy, **eval_kwargs)

    print("Evaluating receding-horizon policy ...")
    rh_results = evaluate_policy(policy=receding_policy, **eval_kwargs)

    # ----------------------------------------------------------- write JSON
    ol_path = os.path.join(out_dir, "eval_open_loop.json")
    rh_path = os.path.join(out_dir, "eval_receding.json")

    with open(ol_path, "w") as fh:
        json.dump(ol_results, fh, indent=2)

    with open(rh_path, "w") as fh:
        json.dump(rh_results, fh, indent=2)

    print(
        f"Open-loop  | success_rate={ol_results['success_rate']:.3f} "
        f"| return_mean={ol_results['return_mean']:.4f} | {ol_path}"
    )
    print(
        f"Receding   | success_rate={rh_results['success_rate']:.3f} "
        f"| return_mean={rh_results['return_mean']:.4f} | {rh_path}"
    )

    return ol_results, rh_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a diffusion checkpoint under both execution modes."
    )
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to checkpoint.pt produced by train_diffusion.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="runs/_smoke/diffusion_seed0",
        help="Directory for eval JSON outputs.",
    )
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--episodes",         type=int,   default=5)
    parser.add_argument("--max_steps",        type=int,   default=50)
    parser.add_argument("--K",                type=int,   default=10,
                        help="DDIM denoising steps.")
    parser.add_argument("--sample_seed_base", type=int,   default=999)
    parser.add_argument("--device",           type=str,   default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diffusion_eval(
        env_id=args.env_id,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        K=args.K,
        sample_seed_base=args.sample_seed_base,
        device=args.device,
    )


if __name__ == "__main__":
    main()
