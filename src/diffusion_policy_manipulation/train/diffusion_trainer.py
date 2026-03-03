"""
Minimal diffusion policy training loop.

train_diffusion
    Trains an MLPDenoiser on pre-recorded action sequences using the
    epsilon-prediction MSE objective.  All randomness (batch sampling,
    timestep draws, noise) is seeded deterministically per step so
    re-running the function produces identical weights and losses.

Outputs written to run_dir
--------------------------
checkpoint.pt       – denoiser state_dict + all hyperparams needed for eval
config.json         – full training configuration
train_summary.json  – final_loss, loss_first, loss_last_50_mean
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.optim as optim

from diffusion_policy_manipulation.data.dataset import NpzTrajectoryDataset
from diffusion_policy_manipulation.models.diffusion_denoiser import MLPDenoiser
from diffusion_policy_manipulation.models.diffusion_schedule import make_linear_schedule
from diffusion_policy_manipulation.train.diffusion_losses import diffusion_eps_loss
from diffusion_policy_manipulation.utils.seeding import set_global_seeds


# ---------------------------------------------------------------------------
# Sequence construction helper
# ---------------------------------------------------------------------------

def _build_sequences(
    dataset: NpzTrajectoryDataset,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build non-episode-crossing action sequences of length *horizon*.

    Parameters
    ----------
    dataset:
        Loaded NpzTrajectoryDataset.
    horizon:
        Sequence length H.

    Returns
    -------
    seq_obs : np.ndarray
        Shape ``[M, obs_dim]`` — observation at the start of each sequence.
    seq_actions : np.ndarray
        Shape ``[M, H, act_dim]`` — consecutive actions within the same episode.
    """
    obs_all = dataset._obs            # [N, obs_dim]
    actions_all = dataset._actions    # [N, act_dim]
    episode_ids = dataset._episode_id # [N]
    N = len(dataset)

    seq_obs_list: list[np.ndarray] = []
    seq_actions_list: list[np.ndarray] = []

    for i in range(N - horizon + 1):
        # Only include sequences that stay within one episode.
        if np.all(episode_ids[i : i + horizon] == episode_ids[i]):
            seq_obs_list.append(obs_all[i])
            seq_actions_list.append(actions_all[i : i + horizon])

    if not seq_obs_list:
        raise RuntimeError(
            f"No valid sequences of length {horizon} found in dataset. "
            "Check that episodes are longer than horizon."
        )

    seq_obs = np.stack(seq_obs_list, axis=0).astype(np.float32)        # [M, obs_dim]
    seq_actions = np.stack(seq_actions_list, axis=0).astype(np.float32) # [M, H, act_dim]
    return seq_obs, seq_actions


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_diffusion(
    dataset_path: str,
    run_dir: str,
    seed: int,
    batch_size: int,
    steps: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    horizon: int,
    t_embed_dim: int,
    T: int,
    beta_start: float,
    beta_end: float,
    device: str,
) -> str:
    """Train a diffusion denoiser on action sequences and save outputs.

    Parameters
    ----------
    dataset_path:
        Path to the .npz file produced by ``record_dataset.py``.
    run_dir:
        Directory where checkpoint, config, and summary are written.
    seed:
        Master seed.  Batch sampling uses ``RandomState(seed + step)``;
        timestep / noise sampling uses a ``torch.Generator`` seeded with
        ``seed + step + 777``.
    batch_size, steps, lr:
        Standard training hyperparameters.
    hidden_dim, num_layers, horizon, t_embed_dim:
        MLPDenoiser architecture hyperparameters.
    T, beta_start, beta_end:
        Linear diffusion schedule parameters.
    device:
        Torch device string.

    Returns
    -------
    str
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(run_dir, exist_ok=True)
    set_global_seeds(seed)

    # ------------------------------------------------------------------ data
    dataset = NpzTrajectoryDataset(dataset_path)
    seq_obs, seq_actions = _build_sequences(dataset, horizon)
    M = len(seq_obs)
    obs_dim = seq_obs.shape[1]
    act_dim = seq_actions.shape[2]

    print(
        f"Sequences: {M}  |  obs_dim={obs_dim}  |  act_dim={act_dim}  "
        f"|  horizon={horizon}"
    )

    # --------------------------------------------------------------- schedule
    dev = torch.device(device)
    schedule = make_linear_schedule(T, beta_start, beta_end, device)

    # ---------------------------------------------------------------- denoiser
    denoiser = MLPDenoiser(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        t_embed_dim=t_embed_dim,
    ).to(dev)

    optimizer = optim.Adam(denoiser.parameters(), lr=lr)

    # --------------------------------------------------------------- training
    loss_history: list[float] = []

    for step in range(steps):
        # --- Deterministic batch selection (NumPy RandomState per step) ---
        rng = np.random.RandomState(seed + step)
        idx = rng.choice(M, size=batch_size, replace=True)

        obs_batch = torch.tensor(seq_obs[idx], dtype=torch.float32, device=dev)
        act_batch = torch.tensor(seq_actions[idx], dtype=torch.float32, device=dev)

        # --- Deterministic timestep + noise (torch Generator per step) ---
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed + step + 777)

        t_batch = torch.randint(0, T, (batch_size,), generator=gen, device=dev)
        noise = torch.randn(
            batch_size, horizon, act_dim, generator=gen, device=dev
        )

        # --- Forward + backward ---
        denoiser.train()
        loss = diffusion_eps_loss(denoiser, schedule, obs_batch, act_batch, t_batch, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))

    final_loss = loss_history[-1]

    # ------------------------------------------------------- save checkpoint
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": denoiser.state_dict(),
            # Schedule params — needed to rebuild schedule at eval time.
            "T": T,
            "beta_start": beta_start,
            "beta_end": beta_end,
            # Denoiser architecture — needed to rebuild model at eval time.
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "t_embed_dim": t_embed_dim,
        },
        checkpoint_path,
    )

    # ----------------------------------------------------------- save config
    config = {
        "dataset_path": dataset_path,
        "run_dir": run_dir,
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "horizon": horizon,
        "t_embed_dim": t_embed_dim,
        "T": T,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "device": device,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "num_sequences": M,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    # --------------------------------------------------------- save summary
    summary: dict = {
        "final_loss": final_loss,
        "loss_first": loss_history[0],
    }
    if steps >= 50:
        summary["loss_last_50_mean"] = float(np.mean(loss_history[-50:]))

    with open(os.path.join(run_dir, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Training complete | steps={steps} | final_loss={final_loss:.6f} | "
        f"checkpoint={checkpoint_path}"
    )
    return os.path.abspath(checkpoint_path)
