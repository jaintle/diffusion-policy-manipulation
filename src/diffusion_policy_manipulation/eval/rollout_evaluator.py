"""
Unified environment rollout evaluator.

evaluate_policy
    Runs N episodes of any Policy-protocol object inside a Gymnasium
    environment and returns a standardised metrics dict.

Determinism guarantee
---------------------
Given a deterministic policy (fixed weights, controlled seeds) and identical
(env_id, seed, episodes, max_steps), two calls produce byte-identical dicts.
This holds because:
  - env resets use fixed per-episode seeds (seed + episode_index)
  - policy.reset() is called with the same per-episode seed each run
  - no global RNG state is consumed between episodes
"""

from __future__ import annotations

import hashlib

import numpy as np

from diffusion_policy_manipulation.envs.make_env import make_env
from diffusion_policy_manipulation.eval.policy_wrappers import Policy


def evaluate_policy(
    env_id: str,
    policy: Policy,
    seed: int,
    episodes: int,
    max_steps: int,
) -> dict:
    """Roll out *policy* for *episodes* episodes and return a metrics dict.

    Parameters
    ----------
    env_id:
        Registered Gymnasium environment ID.
    policy:
        Any object implementing the Policy protocol (reset + act).
    seed:
        Master seed.  Episode *i* resets with ``seed + i``.
    episodes:
        Number of episodes to run.
    max_steps:
        Hard cap on steps per episode.

    Returns
    -------
    dict with keys:
        env_id, seed, episodes,
        episode_returns, episode_lengths, success_flags,
        return_mean, return_std, success_rate,
        eval_seed_list_hash
    """
    env = make_env(env_id, seed=seed)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    success_flags: list[int] = []
    reset_seeds: list[int] = []

    for ep_idx in range(episodes):
        ep_seed = seed + ep_idx
        reset_seeds.append(ep_seed)

        obs, _ = env.reset(seed=ep_seed)
        policy.reset(seed=ep_seed)

        ep_return = 0.0
        ep_length = 0
        terminated = False
        info: dict = {}

        for _ in range(max_steps):
            action = policy.act(np.asarray(obs, dtype=np.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1

            if terminated or truncated:
                break

        # Resolve success from the final info dict or from termination flag.
        if "success" in info:
            ep_success = int(bool(info["success"]))
        elif "is_success" in info:
            ep_success = int(bool(info["is_success"]))
        else:
            ep_success = 1 if terminated else 0

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        success_flags.append(ep_success)

    env.close()

    return_mean = float(np.mean(episode_returns))
    return_std = float(np.std(episode_returns))
    success_rate = float(np.mean(success_flags))
    episode_len_mean = float(np.mean(episode_lengths))

    seeds_str = ",".join(str(s) for s in reset_seeds)
    seeds_hash = hashlib.sha256(seeds_str.encode()).hexdigest()[:16]

    return {
        "env_id": env_id,
        "seed": seed,
        "episodes": episodes,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "success_flags": success_flags,
        "return_mean": return_mean,
        "return_std": return_std,
        "success_rate": success_rate,
        "episode_len_mean": episode_len_mean,
        "eval_seed_list_hash": seeds_hash,
    }
