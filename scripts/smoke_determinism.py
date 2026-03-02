"""
Phase 1 Smoke Test – Determinism Verification
==============================================

Verifies two invariants:

1. Same seed  →  identical trajectories across two independent env instances.
2. Same action sequence  →  identical observations, rewards, and termination
   flags at every step.

Usage
-----
    python scripts/smoke_determinism.py
    python scripts/smoke_determinism.py --env_id gym_pusht/PushT-v0 --seed 42 --steps 100

Prerequisites
-------------
    pip install gym-pusht   # registers gym_pusht/PushT-v0

Exit codes
----------
    0   All assertions passed  ("Determinism smoke test PASSED")
    1   A mismatch was detected (details printed before exit)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# Ensure the src package is importable when the script is executed from the
# repository root without an explicit PYTHONPATH.
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.envs.make_env import make_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify environment determinism across two independent instances."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID (default: gym_pusht/PushT-v0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed (default: 0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps to run (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_id: str = args.env_id
    seed: int = args.seed
    steps: int = args.steps

    # ------------------------------------------------------------------
    # Create two independent environment instances with the same seed.
    # ------------------------------------------------------------------
    env1 = make_env(env_id, seed=seed)
    env2 = make_env(env_id, seed=seed)

    # ------------------------------------------------------------------
    # Reset both environments with the same seed to obtain identical
    # initial observations.
    # ------------------------------------------------------------------
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    obs1 = np.asarray(obs1, dtype=np.float64)
    obs2 = np.asarray(obs2, dtype=np.float64)

    if not np.allclose(obs1, obs2, atol=1e-8):
        print(f"[FAIL] Initial observations differ after reset.")
        print(f"       max |obs1 - obs2| = {np.max(np.abs(obs1 - obs2)):.6e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build a fixed, deterministic action sequence using a local RNG.
    # We do NOT use env.action_space.sample() to avoid any hidden RNG
    # coupling with the environment's internal state.
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed + 12345)

    low = env1.action_space.low
    high = env1.action_space.high

    # ------------------------------------------------------------------
    # Step both envs identically and assert full parity at each step.
    # ------------------------------------------------------------------
    for step_idx in range(steps):
        action = rng.uniform(low, high).astype(np.float32)

        result1 = env1.step(action)
        result2 = env2.step(action)

        obs1, reward1, terminated1, truncated1, _ = result1
        obs2, reward2, terminated2, truncated2, _ = result2

        obs1 = np.asarray(obs1, dtype=np.float64)
        obs2 = np.asarray(obs2, dtype=np.float64)

        obs_ok = np.allclose(obs1, obs2, atol=1e-8)
        rew_ok = abs(reward1 - reward2) <= 1e-10
        term_ok = terminated1 == terminated2
        trunc_ok = truncated1 == truncated2

        if not (obs_ok and rew_ok and term_ok and trunc_ok):
            print(f"[FAIL] Mismatch detected at step {step_idx}.")
            if not obs_ok:
                print(
                    f"       Observation: max |obs1 - obs2| = "
                    f"{np.max(np.abs(obs1 - obs2)):.6e}"
                )
            if not rew_ok:
                print(
                    f"       Reward:      reward1={reward1:.10f}  "
                    f"reward2={reward2:.10f}"
                )
            if not term_ok:
                print(
                    f"       Terminated:  terminated1={terminated1}  "
                    f"terminated2={terminated2}"
                )
            if not trunc_ok:
                print(
                    f"       Truncated:   truncated1={truncated1}  "
                    f"truncated2={truncated2}"
                )
            env1.close()
            env2.close()
            sys.exit(1)

        # Stop stepping once both episodes are done (avoids stepping a
        # terminated env, which some wrappers do not support).
        if terminated1 or truncated1:
            break

    env1.close()
    env2.close()

    print("Determinism smoke test PASSED")


if __name__ == "__main__":
    main()
