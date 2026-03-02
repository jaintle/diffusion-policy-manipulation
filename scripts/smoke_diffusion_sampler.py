"""
Phase 4 Smoke Test – Diffusion Sampler Determinism
===================================================

Verifies:
1. The diffusion schedule, embeddings, denoiser, and sampler run end-to-end
   without crashing.
2. The output shape is correct: [B, horizon, act_dim].
3. Two calls to sample_ddim with the same seed produce exactly identical
   tensors (torch.equal).

Usage
-----
    python scripts/smoke_diffusion_sampler.py
    python scripts/smoke_diffusion_sampler.py --seed 42 --B 2 --steps 5

Exit codes
----------
    0   All checks passed  ("Diffusion sampler smoke test PASSED")
    1   A check failed (details printed before exit)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.models.diffusion_denoiser import MLPDenoiser   # noqa: E402
from diffusion_policy_manipulation.models.diffusion_schedule import (              # noqa: E402
    make_linear_schedule,
)
from diffusion_policy_manipulation.models.samplers import sample_ddim              # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify determinism of the DDIM diffusion sampler."
    )
    parser.add_argument("--seed",    type=int, default=0,  help="Master seed.")
    parser.add_argument("--B",       type=int, default=4,  help="Batch size.")
    parser.add_argument("--obs_dim", type=int, default=10, help="Observation dim.")
    parser.add_argument("--act_dim", type=int, default=2,  help="Action dim.")
    parser.add_argument("--horizon", type=int, default=8,  help="Action horizon H.")
    parser.add_argument("--T",       type=int, default=50, help="Diffusion steps T.")
    parser.add_argument("--steps",   type=int, default=10, help="DDIM steps.")
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    seed    = args.seed
    B       = args.B
    obs_dim = args.obs_dim
    act_dim = args.act_dim
    horizon = args.horizon
    T       = args.T
    steps   = args.steps

    device = "cpu"

    # ------------------------------------------------------------------
    # A) Build a deterministic dummy observation batch.
    # ------------------------------------------------------------------
    gen_obs = torch.Generator(device=device)
    gen_obs.manual_seed(seed + 111)
    obs = torch.randn(B, obs_dim, generator=gen_obs, device=device)

    # ------------------------------------------------------------------
    # B) Instantiate schedule and denoiser with fixed hyperparams.
    # ------------------------------------------------------------------
    schedule = make_linear_schedule(
        T=T,
        beta_start=1e-4,
        beta_end=2e-2,
        device=device,
    )

    denoiser = MLPDenoiser(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=horizon,
        hidden_dim=128,
        num_layers=2,
        t_embed_dim=64,
    )
    denoiser.eval()

    # ------------------------------------------------------------------
    # C) Sample twice with the same seed.
    # ------------------------------------------------------------------
    sample_seed = seed + 999

    out1 = sample_ddim(
        denoiser=denoiser,
        schedule=schedule,
        obs=obs,
        horizon=horizon,
        act_dim=act_dim,
        steps=steps,
        seed=sample_seed,
    )

    out2 = sample_ddim(
        denoiser=denoiser,
        schedule=schedule,
        obs=obs,
        horizon=horizon,
        act_dim=act_dim,
        steps=steps,
        seed=sample_seed,
    )

    # ------------------------------------------------------------------
    # D) Assert exact equality.
    # ------------------------------------------------------------------
    if not torch.equal(out1, out2):
        max_diff = (out1 - out2).abs().max().item()
        _fail(
            f"Sampler non-determinism detected.  max|out1 - out2| = {max_diff:.6e}"
        )

    # ------------------------------------------------------------------
    # E) Assert correct output shape.
    # ------------------------------------------------------------------
    expected_shape = (B, horizon, act_dim)
    if tuple(out1.shape) != expected_shape:
        _fail(
            f"Output shape mismatch: got {tuple(out1.shape)}, "
            f"expected {expected_shape}"
        )

    print("Diffusion sampler smoke test PASSED")


if __name__ == "__main__":
    main()
