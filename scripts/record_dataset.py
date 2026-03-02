"""
Record a deterministic offline dataset from a Gymnasium environment.

Rolls out the environment using a seeded action RNG and saves all transitions
to a single .npz file.  No model, no normalization — raw transitions only.

Usage
-----
    python scripts/record_dataset.py
    python scripts/record_dataset.py --env_id gym_pusht/PushT-v0 --seed 0 \
        --episodes 10 --max_steps 200 --out data/pusht_demo_seed0.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make the src package importable when run from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_policy_manipulation.envs.make_env import make_env  # noqa: E402


def record(
    env_id: str,
    seed: int,
    episodes: int,
    max_steps: int,
    out: str,
) -> None:
    """Roll out the environment deterministically and save transitions to *out*.

    Parameters
    ----------
    env_id:
        Registered Gymnasium environment identifier.
    seed:
        Master seed.  Action RNG is seeded with ``seed + 12345``; each episode
        resets with ``seed + episode_index``.
    episodes:
        Number of episodes to record.
    max_steps:
        Maximum timesteps per episode (episode may end earlier on termination /
        truncation).
    out:
        Output path for the .npz file.  Parent directories are created
        automatically.
    """
    # Ensure output directory exists (mkdir -p behaviour).
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(env_id, seed=seed)

    # Deterministic action RNG — shared across all episodes so the full
    # sequence is reproducible from a single seed.
    rng = np.random.RandomState(seed + 12345)
    low = env.action_space.low
    high = env.action_space.high

    # Per-transition accumulators.
    obs_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []
    next_obs_list: list[np.ndarray] = []
    episode_id_list: list[int] = []
    timestep_list: list[int] = []

    for ep_idx in range(episodes):
        obs, _ = env.reset(seed=seed + ep_idx)
        obs = np.asarray(obs, dtype=np.float32)

        for step_idx in range(max_steps):
            action = rng.uniform(low, high).astype(np.float32)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))
            next_obs_list.append(next_obs)
            episode_id_list.append(ep_idx)
            timestep_list.append(step_idx)

            obs = next_obs

            if terminated or truncated:
                break

    env.close()

    # Stack into contiguous arrays.
    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)         # [N, obs_dim]
    actions_arr = np.stack(actions_list, axis=0).astype(np.float32) # [N, act_dim]
    rewards_arr = np.array(rewards_list, dtype=np.float32)          # [N]
    terminated_arr = np.array(terminated_list, dtype=np.uint8)      # [N]
    truncated_arr = np.array(truncated_list, dtype=np.uint8)        # [N]
    next_obs_arr = np.stack(next_obs_list, axis=0).astype(np.float32) # [N, obs_dim]
    episode_id_arr = np.array(episode_id_list, dtype=np.int32)      # [N]
    timestep_arr = np.array(timestep_list, dtype=np.int32)          # [N]

    np.savez(
        out,
        obs=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        terminated=terminated_arr,
        truncated=truncated_arr,
        next_obs=next_obs_arr,
        episode_id=episode_id_arr,
        timestep=timestep_arr,
    )

    N = len(obs_list)
    obs_dim = obs_arr.shape[1]
    act_dim = actions_arr.shape[1]
    print(
        f"Recorded {N} transitions | obs_dim={obs_dim} | act_dim={act_dim} | saved to {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record deterministic rollouts to a .npz dataset file."
    )
    parser.add_argument(
        "--env_id", type=str, default="gym_pusht/PushT-v0",
        help="Registered Gymnasium environment ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master seed (default: 0).",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to record (default: 10).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=200,
        help="Maximum steps per episode (default: 200).",
    )
    parser.add_argument(
        "--out", type=str, default="data/pusht_demo_seed0.npz",
        help="Output .npz path (default: data/pusht_demo_seed0.npz).",
    )
    args = parser.parse_args()

    record(
        env_id=args.env_id,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        out=args.out,
    )


if __name__ == "__main__":
    main()
