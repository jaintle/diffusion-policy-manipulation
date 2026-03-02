"""
Phase 5 Smoke Test – Execution Strategy Determinism
====================================================

Verifies:
1. Gaussian MLP BC, diffusion open-loop, and diffusion receding-horizon
   wrappers all complete rollouts without crashing.
2. Diffusion open-loop evaluation is deterministic:
   two runs with identical seeds return identical episode_returns and metrics.
3. Diffusion receding-horizon evaluation is deterministic under the same
   conditions.

Usage
-----
    python scripts/smoke_execution_modes.py
    python scripts/smoke_execution_modes.py --env_id gym_pusht/PushT-v0 --seed 0

Exit codes
----------
    0   All checks passed  ("Execution modes smoke test PASSED")
    1   A mismatch or crash was detected (details printed before exit)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_SRC, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusion_policy_manipulation.envs.make_env import make_env                           # noqa: E402
from diffusion_policy_manipulation.eval.policy_wrappers import (                           # noqa: E402
    DiffusionOpenLoopPolicyWrapper,
    DiffusionRecedingHorizonPolicyWrapper,
    GaussianBCPolicyWrapper,
)
from diffusion_policy_manipulation.eval.rollout_evaluator import evaluate_policy           # noqa: E402
from diffusion_policy_manipulation.models.diffusion_denoiser import MLPDenoiser            # noqa: E402
from diffusion_policy_manipulation.models.diffusion_schedule import make_linear_schedule   # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 execution modes smoke test.")
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master seed (default: 0).",
    )
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _assert_dicts_equal(a: dict, b: dict, label: str) -> None:
    """Deep-compare two evaluation result dicts; fail with details on mismatch."""
    for key in a:
        if key not in b:
            _fail(f"{label}: key '{key}' in run_a missing from run_b.")
        va, vb = a[key], b[key]
        if isinstance(va, list):
            if va != vb:
                _fail(
                    f"{label}: list mismatch at key='{key}'\n"
                    f"  run_a: {va}\n"
                    f"  run_b: {vb}"
                )
        elif isinstance(va, float):
            if va != vb:
                _fail(
                    f"{label}: float mismatch at key='{key}' "
                    f"run_a={va} run_b={vb}"
                )
        else:
            if va != vb:
                _fail(
                    f"{label}: value mismatch at key='{key}' "
                    f"run_a={va!r} run_b={vb!r}"
                )
    for key in b:
        if key not in a:
            _fail(f"{label}: key '{key}' in run_b missing from run_a.")


def main() -> None:
    args = parse_args()
    env_id = args.env_id
    seed   = args.seed
    device = "cpu"

    # ------------------------------------------------------------------
    # A) Ensure BC checkpoint exists (produced by Phase 3 smoke).
    # ------------------------------------------------------------------
    ckpt_path = f"runs/_smoke/bc_seed{seed}/checkpoint.pt"

    if not os.path.isfile(ckpt_path):
        print(f"BC checkpoint not found at {ckpt_path} — running smoke_bc.py ...")
        smoke_bc = os.path.join(_SCRIPTS, "smoke_bc.py")
        result = subprocess.run(
            [sys.executable, smoke_bc, "--env_id", env_id, "--seed", str(seed)],
            check=False,
        )
        if result.returncode != 0:
            _fail(f"smoke_bc.py exited with code {result.returncode}.")
        if not os.path.isfile(ckpt_path):
            _fail(f"smoke_bc.py ran but checkpoint still missing: {ckpt_path}")
    else:
        print(f"Reusing existing BC checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # B) Instantiate BC policy wrapper.
    # ------------------------------------------------------------------
    bc_policy = GaussianBCPolicyWrapper(checkpoint_path=ckpt_path, device=device)

    # ------------------------------------------------------------------
    # C) Infer obs_dim / act_dim from the environment.
    # ------------------------------------------------------------------
    _probe_env = make_env(env_id, seed=seed)
    obs_dim = _probe_env.observation_space.shape[0]
    act_dim = _probe_env.action_space.shape[0]
    _probe_env.close()

    print(f"env obs_dim={obs_dim}  act_dim={act_dim}")

    # ------------------------------------------------------------------
    #    Diffusion components (random weights — determinism still holds).
    # ------------------------------------------------------------------
    HORIZON    = 8
    T          = 50
    DDIM_STEPS = 10

    schedule = make_linear_schedule(
        T=T, beta_start=1e-4, beta_end=0.02, device=device
    )

    denoiser = MLPDenoiser(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=HORIZON,
        hidden_dim=128,
        num_layers=2,
        t_embed_dim=64,
    )
    denoiser.eval()

    # ------------------------------------------------------------------
    # D) Create both diffusion wrappers.
    # ------------------------------------------------------------------
    open_loop_policy = DiffusionOpenLoopPolicyWrapper(
        denoiser=denoiser,
        schedule=schedule,
        horizon=HORIZON,
        act_dim=act_dim,
        ddim_steps=DDIM_STEPS,
        sample_seed_base=100_000,
        device=device,
    )

    receding_policy = DiffusionRecedingHorizonPolicyWrapper(
        denoiser=denoiser,
        schedule=schedule,
        horizon=HORIZON,
        act_dim=act_dim,
        ddim_steps=DDIM_STEPS,
        sample_seed_base=200_000,
        device=device,
    )

    # ------------------------------------------------------------------
    # E) Run evaluate_policy for all three policies.
    # ------------------------------------------------------------------
    EVAL_EPISODES = 3
    EVAL_MAX_STEPS = 50

    eval_kwargs = dict(
        env_id=env_id,
        seed=seed,
        episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
    )

    print("\n--- BC (mean action) ---")
    bc_results = evaluate_policy(policy=bc_policy, **eval_kwargs)

    print("\n--- Diffusion open-loop ---")
    ol_results_a = evaluate_policy(policy=open_loop_policy, **eval_kwargs)

    print("\n--- Diffusion receding-horizon ---")
    rh_results_a = evaluate_policy(policy=receding_policy, **eval_kwargs)

    # ------------------------------------------------------------------
    # F) Determinism checks — run each diffusion policy a second time.
    # ------------------------------------------------------------------
    print("\n--- Determinism re-runs ---")
    ol_results_b = evaluate_policy(policy=open_loop_policy, **eval_kwargs)
    rh_results_b = evaluate_policy(policy=receding_policy, **eval_kwargs)

    _assert_dicts_equal(ol_results_a, ol_results_b, "diffusion open-loop")
    _assert_dicts_equal(rh_results_a, rh_results_b, "diffusion receding-horizon")

    # ------------------------------------------------------------------
    # G) Summary.
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(
        f"BC            | success_rate={bc_results['success_rate']:.3f} "
        f"| return_mean={bc_results['return_mean']:.4f}"
    )
    print(
        f"Open-loop     | success_rate={ol_results_a['success_rate']:.3f} "
        f"| return_mean={ol_results_a['return_mean']:.4f}"
    )
    print(
        f"Receding-horiz| success_rate={rh_results_a['success_rate']:.3f} "
        f"| return_mean={rh_results_a['return_mean']:.4f}"
    )

    print("\nExecution modes smoke test PASSED")


if __name__ == "__main__":
    main()
