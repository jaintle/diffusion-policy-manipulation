"""
Minimal Behavior Cloning training loop.

train_bc
    Trains a GaussianMLPPolicy on a pre-recorded .npz dataset using
    Gaussian NLL supervision.  No framework dependency beyond PyTorch.

Outputs written to run_dir
--------------------------
checkpoint.pt       – model state_dict + metadata
config.json         – full hyperparameter record
train_summary.json  – final_loss, loss_first, loss_last_100_mean
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.optim as optim

from diffusion_policy_manipulation.data.dataset import NpzTrajectoryDataset
from diffusion_policy_manipulation.models.mlp_bc import GaussianMLPPolicy
from diffusion_policy_manipulation.train.bc_losses import gaussian_nll
from diffusion_policy_manipulation.utils.seeding import set_global_seeds


def train_bc(
    dataset_path: str,
    run_dir: str,
    seed: int,
    batch_size: int,
    steps: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    device: str,
) -> str:
    """Train a Gaussian MLP BC policy and save outputs to *run_dir*.

    Parameters
    ----------
    dataset_path:
        Path to the .npz file produced by ``record_dataset.py``.
    run_dir:
        Directory where checkpoint, config, and summary are written.
    seed:
        Master seed for reproducibility.
    batch_size:
        Transitions sampled per gradient step.
    steps:
        Total number of gradient steps.
    lr:
        Adam learning rate.
    hidden_dim:
        Width of each hidden layer in the MLP.
    num_layers:
        Number of hidden layers.
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    str
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(run_dir, exist_ok=True)
    set_global_seeds(seed)

    # ------------------------------------------------------------------ data
    dataset = NpzTrajectoryDataset(dataset_path)
    obs_dim = dataset._obs.shape[1]
    act_dim = dataset._actions.shape[1]

    # ---------------------------------------------------------------- model
    dev = torch.device(device)
    policy = GaussianMLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # --------------------------------------------------------------- training
    loss_history: list[float] = []

    for step in range(steps):
        # Deterministic batch per step — same (step, seed) always yields
        # the same indices regardless of run order.
        batch = dataset.sample_batch(batch_size, seed=seed + step)

        obs_t = torch.tensor(batch["obs"], dtype=torch.float32, device=dev)
        act_t = torch.tensor(batch["actions"], dtype=torch.float32, device=dev)

        mean, log_std = policy(obs_t)
        loss = gaussian_nll(mean, log_std, act_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.detach().cpu()))

    final_loss = loss_history[-1]

    # ------------------------------------------------------- save checkpoint
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seed": seed,
            "steps": steps,
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
        "device": device,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    # --------------------------------------------------------- save summary
    summary: dict = {
        "final_loss": final_loss,
        "loss_first": loss_history[0],
    }
    if steps >= 100:
        summary["loss_last_100_mean"] = float(np.mean(loss_history[-100:]))

    with open(os.path.join(run_dir, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Training complete | steps={steps} | final_loss={final_loss:.6f} | "
        f"checkpoint={checkpoint_path}"
    )
    return os.path.abspath(checkpoint_path)
