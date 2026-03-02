"""
Deterministic environment factory for Gymnasium environments.

make_env creates an environment, applies global and space seeds,
and performs an initial seeded reset so every subsequent episode
can be started from a reproducible state.
"""

from __future__ import annotations

import gymnasium as gym

# gym_pusht registers "gym_pusht/PushT-v0" only when its __init__ is imported.
# Gymnasium does not auto-discover third-party packages, so we import it here
# to trigger registration before any gym.make() call.  The import is silently
# skipped when the package is absent so other environments still work.
try:
    import gym_pusht  # noqa: F401
except ModuleNotFoundError:
    pass

from diffusion_policy_manipulation.utils.seeding import seed_env_spaces, set_global_seeds


def make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
) -> gym.Env:
    """Create and seed a Gymnasium environment.

    Parameters
    ----------
    env_id:
        Registered Gymnasium environment identifier, e.g. ``"gym_pusht/PushT-v0"``.
    seed:
        Non-negative integer seed.  Applied to global RNGs, the environment
        reset, and both action/observation spaces.
    render_mode:
        Optional render mode passed to ``gym.make``.  When ``None`` the
        environment is created without a render mode argument.

    Returns
    -------
    gym.Env
        A seeded, ready-to-use environment instance.
    """
    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

    set_global_seeds(seed)
    env.reset(seed=seed)
    seed_env_spaces(env, seed)

    return env
