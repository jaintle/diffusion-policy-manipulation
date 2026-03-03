# Experimental Protocol

This document describes the exact protocol used for dataset generation, model training,
evaluation, and multi-seed aggregation. Any deviation from these parameters constitutes
a different experiment and should be recorded as such in `report/experiment_log.md`.

---

## 1. Dataset Generation

**Script:** `scripts/record_dataset.py`

**Environment:** `gym_pusht/PushT-v0` (Push-T, low-dimensional state)

**Observation space:** 5-dimensional continuous vector (agent position x, y; block
position x, y; block angle).

**Action space:** 2-dimensional continuous vector (target position x, y), clipped to
the environment range.

**Collection policy:** Uniform random actions drawn from
`np.random.RandomState(seed + 12345)`. This is not an expert policy. The dataset covers
a broad but sparse region of the state-action space.

**Episode seeding:** Episode `i` resets the environment with seed `dataset_seed + i`.

**Parameters (default for smoke / recommended for full run):**

| Parameter         | Smoke | Full run |
|-------------------|-------|----------|
| `--episodes`      | 3     | 200      |
| `--max_steps`     | 50    | 300      |
| `--seed`          | 0     | 0        |

**Output format:** NumPy `.npz` archive with the following arrays:

| Key           | Shape                   | Dtype   | Description                      |
|---------------|-------------------------|---------|----------------------------------|
| `obs`         | `[N, obs_dim]`          | float32 | Observations at each timestep    |
| `next_obs`    | `[N, obs_dim]`          | float32 | Next observations                |
| `actions`     | `[N, act_dim]`          | float32 | Actions taken                    |
| `rewards`     | `[N]`                   | float32 | Per-step reward                  |
| `terminated`  | `[N]`                   | uint8   | Episode terminated flag          |
| `truncated`   | `[N]`                   | uint8   | Episode truncated flag           |
| `episode_id`  | `[N]`                   | int32   | Episode index for each timestep  |
| `timestep`    | `[N]`                   | int32   | Within-episode step index        |

`N` is the total number of transitions across all episodes. Episodes are stored
contiguously in chronological order.

**Normalization:** A `RunningNormalizer` is fit on the recorded dataset and saved
alongside the `.npz` file as a JSON sidecar. The normalizer computes per-dimension
mean and standard deviation over all transitions. Standard deviation is clipped to a
minimum of `1e-6` to avoid division by zero in degenerate dimensions. Both BC and
diffusion training consume normalized observations and actions.

---

## 2. Gaussian MLP BC Training

**Script:** `scripts/train_bc.py`

**Model:** `GaussianMLPPolicy`

- MLP backbone: `[hidden_dim] * num_layers` with ReLU activations.
- Output: mean vector `[act_dim]` from a linear head; a learned scalar log-standard
  deviation parameter (single `nn.Parameter`, shared across all action dimensions).
- At deterministic evaluation time, only the mean is used.

**Loss:** Gaussian negative log-likelihood.
`NLL = 0.5 * log(2π) + log_std + 0.5 * ((action - mean) / exp(log_std))^2`
Log standard deviation is clamped to `[-20, 2]` before exponentiation.

**Optimizer:** Adam, default betas.

**Batch construction:** At step `t`, a fresh `np.random.RandomState(seed + t)` draws a
batch of indices with replacement from the full transition pool. No persistent RNG state
is shared between steps.

**Default architecture:**

| Hyperparameter | Default |
|----------------|---------|
| `hidden_dim`   | 256     |
| `num_layers`   | 3       |
| `lr`           | 3e-4    |
| `batch_size`   | 256     |

**Outputs written to `--run_dir`:**

- `checkpoint.pt` — model weights and architecture config.
- `config.json` — full training configuration.
- `train_summary.json` — final training loss.

---

## 3. Diffusion Policy Training

**Script:** `scripts/train_diffusion.py`

**Model:** `MLPDenoiser`

- Input: concatenation of `[obs, x_t_flat, relu(t_proj(t_embed))]` where `x_t_flat`
  is the noisy action sequence flattened to `[horizon * act_dim]`.
- Timestep embedding: sinusoidal embedding of dimension `t_embed_dim`, projected to
  `hidden_dim` by a learned linear layer.
- Output: predicted noise `eps_hat` reshaped to `[B, horizon, act_dim]`.

**Noise schedule:** Linear beta schedule.
`beta_t = linspace(beta_start, beta_end, T)`
`alpha_bar_t = cumprod(1 - beta_t)`

**Prediction target:** Noise `eps` (epsilon parameterization).

**Training objective:** Mean squared error between predicted and actual noise.
`L = MSE(eps_hat, eps)`
where `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`.

**Action sequences:** Constructed from the dataset by sliding a window of length `H`
over each episode. Sequences that cross episode boundaries are excluded.

**Noise and timestep sampling per step:** A `torch.Generator(seed + step + 777)` is
created fresh at each step and used to draw both the diffusion timestep `t` (uniform
over `[0, T-1]`) and the noise `eps ~ N(0, I)`.

**Default architecture and schedule:**

| Hyperparameter  | Default |
|-----------------|---------|
| `hidden_dim`    | 256     |
| `num_layers`    | 4       |
| `t_embed_dim`   | 64      |
| `horizon H`     | 8       |
| `T`             | 100     |
| `beta_start`    | 1e-4    |
| `beta_end`      | 0.02    |
| `lr`            | 3e-4    |
| `batch_size`    | 256     |

**Outputs written to `--run_dir`:**

- `checkpoint.pt` — model weights and full architecture + schedule config.
- `config.json` — full training configuration.
- `train_summary.json` — final training loss.

---

## 4. Evaluation Protocol

**Scripts:** `scripts/eval_bc.py`, `scripts/eval_diffusion.py`

**Environment:** Same as dataset collection (`gym_pusht/PushT-v0`).

**Episode seeding:** Episode `i` uses reset seed `eval_seed + i`. The master eval seed
is set independently from the training seed.

**Steps per episode:** Capped at `max_steps`. Episodes end earlier on termination or
truncation.

**Success metric:** The environment's `terminated` flag is used as the primary success
signal. If the final `info` dict contains a `"success"` or `"is_success"` key, that
value takes precedence. No reward threshold is used as a success proxy.

**BC evaluation:** At each step, the policy computes the mean action (no sampling). The
log standard deviation parameter is not used during evaluation.

**Diffusion evaluation — Open-loop chunk execution:**

1. On `policy.reset(episode_seed)`, the action cache is cleared.
2. On the first call to `policy.act(obs)` after reset or cache exhaustion:
   - DDIM is called with `seed = episode_seed + chunk_index + sample_seed_base`.
   - The full H-step sequence is stored in the cache.
3. Actions are returned from the cache in order (index 0 through H-1).
4. When the cache is exhausted, step 2 is repeated with `chunk_index += 1`.

**Diffusion evaluation — Receding-horizon re-planning:**

1. On `policy.reset(episode_seed)`, the timestep counter is reset to 0.
2. At every call to `policy.act(obs)`:
   - DDIM is called with `seed = episode_seed + timestep_index + sample_seed_base`.
   - Only action `seq[0, 0]` (first action of the predicted sequence) is returned.
   - The remainder of the sequence is discarded.
3. `timestep_index` is incremented after each call.

**DDIM sampler configuration:**

| Parameter    | Value             |
|--------------|-------------------|
| `eta`        | 0.0 (deterministic) |
| `steps K`    | 10                |
| Timesteps    | `linspace(0, T-1, K)` reversed (large to small) |

**Evaluation output JSON:**

Each output file contains:
- `env_id`, `seed`, `episodes`
- `episode_returns` — list of per-episode cumulative rewards.
- `episode_lengths` — list of per-episode step counts.
- `success_flags` — list of per-episode success indicators (0 or 1).
- `return_mean`, `return_std`, `success_rate`, `episode_len_mean` — scalar summaries.
- `eval_seed_list_hash` — SHA-256 (first 16 hex chars) of the comma-joined reset seed
  list, for reproducibility verification.

---

## 5. Multi-seed Aggregation

**Scripts:** `scripts/reproduce_multiseed.py`, `scripts/validate_results.py`,
`scripts/aggregate_results.py`

**Required file layout before aggregation:**

```
results/{results_root}/
    seed0/
        bc_eval.json
        diff_open_loop.json
        diff_receding.json
    seed1/
        ...
    seed2/
        ...
```

**Validation:** `validate_results.py` checks that all three JSON files are present for
each seed, are parseable, and contain the four required scalar keys
(`success_rate`, `return_mean`, `return_std`, `episode_len_mean`). Missing files or
missing keys cause a hard exit with a descriptive error. Aggregation must never proceed
on incomplete data.

**Aggregation:** `aggregate_results.py` reads the three method files for each seed and
computes per-metric mean and population standard deviation (ddof=0) across seeds.

**Output files:**

`per_seed.csv` — one row per (method, seed) combination; all numeric fields.

`summary.csv` — one row per (method, metric) combination; columns: `method`, `metric`,
`mean`, `std`, `n_seeds`.

Methods reported: `bc`, `diff_open_loop`, `diff_receding`.

Metrics reported: `success_rate`, `return_mean`, `return_std`, `episode_len_mean`,
`mean_policy_time` (if present).

**Missing seeds:** If any seed directory or required file is absent, `aggregate_results.py`
exits with code 1. Partial aggregation is not performed.

---

## 6. Experiment Logging

After each meaningful run, append an entry to `report/experiment_log.md` using the
template provided in that file. Each entry must record the commit hash, seed list,
config parameters, and all quantitative results. Entries are append-only; do not edit
prior entries.
