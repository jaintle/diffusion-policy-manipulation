"""
Deterministic seeding utilities.

set_global_seeds  – seeds Python, NumPy, and PyTorch.
seed_env_spaces   – seeds a Gymnasium environment's action/observation spaces.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed every relevant RNG for reproducible behaviour.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic cuDNN kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Request fully deterministic algorithms where available.
    # Some ops do not have deterministic implementations; we catch the error
    # and emit a short warning instead of crashing so CPU-only runs proceed.
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        print(f"[seeding] torch.use_deterministic_algorithms(True) skipped: {exc}")


def seed_env_spaces(env, seed: int) -> None:
    """Seed the action and observation spaces of a Gymnasium environment.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment instance.
    seed:
        Non-negative integer seed value.
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
