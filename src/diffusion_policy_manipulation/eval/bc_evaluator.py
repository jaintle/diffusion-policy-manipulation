"""
Deterministic evaluation harness for the Gaussian MLP BC policy.

evaluate_bc
    Loads a checkpoint, runs N episodes using the policy's mean action
    (no sampling), and writes a JSON results file.

Determinism guarantee
---------------------
Given identical (checkpoint, seed, episodes, max_steps), two calls to
evaluate_bc produce byte-identical JSON output.  This holds because:
  - environment reset uses a fixed per-episode seed
  - actions are always the deterministic mean (no RNG at inference)
  - JSON serialization order is fixed
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from diffusion_policy_manipulation.envs.make_env import make_env
from diffusion_policy_manipulation.models.mlp_bc import GaussianMLPPolicy
from diffusion_policy_manipulation.utils.seeding import set_global_seeds


def evaluate_bc(
    env_id: str,
    checkpoint_path: str,
    seed: int,
    episodes: int,
    max_steps: int,
    out_path: str,
    device: str,
) -> None:
    """Evaluate a BC checkpoint deterministically and save results to JSON.

    Parameters
    ----------
    env_id:
        Registered Gymnasium environment ID.
    checkpoint_path:
        Path to the .pt checkpoint produced by ``train_bc``.
    seed:
        Master evaluation seed.  Episode *i* resets with ``seed + i``.
    episodes:
        Number of evaluation episodes to run.
    max_steps:
        Maximum steps per episode.
    out_path:
        Path for the output JSON file.  Parent dirs are created automatically.
    device:
        Torch device string.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    set_global_seeds(seed)

    # ---------------------------------------------------------- load policy
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)

    policy = GaussianMLPPolicy(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
    ).to(dev)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    # --------------------------------------------------------- create env
    env = make_env(env_id, seed=seed)

    # --------------------------------------------------------- eval loop
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    success_flags: list[int] = []
    reset_seeds: list[int] = []

    for ep_idx in range(episodes):
        reset_seed = seed + ep_idx
        reset_seeds.append(reset_seed)

        obs, _ = env.reset(seed=reset_seed)

        ep_return = 0.0
        ep_length = 0
        ep_terminated = False

        for _ in range(max_steps):
            obs_t = torch.tensor(
                np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=dev
            ).unsqueeze(0)  # [1, obs_dim]

            with torch.no_grad():
                action = policy.deterministic_action(obs_t)  # [1, act_dim]

            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action_np)

            ep_return += float(reward)
            ep_length += 1
            ep_terminated = bool(terminated)

            if terminated or truncated:
                break

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        # Success heuristic: episode ended via termination (task solved).
        # Environments that expose an explicit "success" info key can be
        # extended here in a later phase.
        success_flags.append(1 if ep_terminated else 0)

    env.close()

    # ---------------------------------------------------- compute summary
    return_mean = float(np.mean(episode_returns))
    return_std = float(np.std(episode_returns))
    success_rate = float(np.mean(success_flags))

    # Deterministic hash of the reset seed list — lets callers verify that
    # two eval runs used the same episode seeds without comparing every array.
    seeds_str = ",".join(str(s) for s in reset_seeds)
    seeds_hash = hashlib.sha256(seeds_str.encode()).hexdigest()[:16]

    results = {
        "env_id": env_id,
        "seed": seed,
        "episodes": episodes,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "success_flags": success_flags,
        "return_mean": return_mean,
        "return_std": return_std,
        "success_rate": success_rate,
        "eval_seed_list_hash": seeds_hash,
    }

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(
        f"Eval | success_rate={success_rate:.3f} | "
        f"return_mean={return_mean:.4f} | return_std={return_std:.4f} | "
        f"saved to {out_path}"
    )
