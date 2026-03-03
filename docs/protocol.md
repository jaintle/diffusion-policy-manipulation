# Experimental Protocol

This document describes the exact protocol used for dataset generation, model training,
evaluation, and multi-seed aggregation. Any deviation from these parameters constitutes
a different experiment and should be recorded as such in `report/experiment_log.md`.

The configuration below matches the run reported in `results/rq_exec_mode/` (commit
04defe5, seeds 0–2, logged in `report/experiment_log.md` as `rq-exec-mode-3seed`).

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

**Parameters used for `results/rq_exec_mode/`:**

| Parameter         | Value used |
|-------------------|------------|
| `--episodes`      | 20         |
| `--max_steps`     | 200        |
| `--seed`          | per-seed (0, 1, 2) |

**Exact CLI command (seed 0; repeat for seeds 1, 2):**

```bash
python scripts/record_dataset.py \
    --env_id    gym_pusht/PushT-v0 \
    --seed      0 \
    --episodes  20 \
    --max_steps 200 \
    --out       data/_repro/pusht_repro_seed0.npz
```

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
minimum of `1e-6`. Both BC and diffusion training consume normalized observations and
actions.

---

## 2. Gaussian MLP BC Training

**Script:** `scripts/train_bc.py`

**Model:** `GaussianMLPPolicy`

- MLP backbone: `[hidden_dim] * num_layers` with ReLU activations.
- Output: mean vector `[act_dim]` from a linear head; a learned scalar log-standard
  deviation parameter (single `nn.Parameter`, shared across all action dimensions).
- At evaluation time, only the mean is used (deterministic action).

**Loss:** Gaussian negative log-likelihood.
`NLL = 0.5 * log(2π) + log_std + 0.5 * ((action - mean) / exp(log_std))^2`
Log standard deviation is clamped to `[-20, 2]` before exponentiation.

**Optimizer:** Adam, default betas.

**Parameters used for `results/rq_exec_mode/`:**

| Hyperparameter | Value  |
|----------------|--------|
| `hidden_dim`   | 256    |
| `num_layers`   | 3      |
| `lr`           | 3e-4   |
| `batch_size`   | 256    |
| `--steps`      | 3 000  |

**Exact CLI command (seed 0):**

```bash
python scripts/train_bc.py \
    --dataset_path data/_repro/pusht_repro_seed0.npz \
    --run_dir      runs/_repro/bc_seed0 \
    --seed         0 \
    --steps        3000 \
    --device       cpu
```

**Outputs written to `--run_dir`:**

- `checkpoint.pt` — model weights and architecture config.
- `config.json` — full training configuration.
- `train_summary.json` — final training loss.

---

## 3. Diffusion Policy Training

**Script:** `scripts/train_diffusion.py`

**Model:** `MLPDenoiser`

- Input: concatenation of `[obs, x_t_flat, relu(t_proj(t_embed))]`.
- Timestep embedding: sinusoidal embedding, projected to `hidden_dim` by a learned
  linear layer.
- Output: predicted noise `eps_hat` reshaped to `[B, horizon, act_dim]`.

**Noise schedule:** Linear beta schedule.
`beta_t = linspace(beta_start, beta_end, T)`; `alpha_bar_t = cumprod(1 - beta_t)`.

**Prediction target:** Noise ε.

**Training objective:** `L = MSE(eps_hat, eps)` where
`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`.

**Parameters used for `results/rq_exec_mode/`:**

| Hyperparameter    | Value  |
|-------------------|--------|
| `hidden_dim`      | 256    |
| `num_layers`      | 4      |
| `t_embed_dim`     | 64     |
| `horizon H`       | 8      |
| `T`               | 50     |
| `beta_start`      | 1e-4   |
| `beta_end`        | 0.02   |
| `lr`              | 3e-4   |
| `batch_size`      | 256    |
| `--steps`         | 5 000  |

**Exact CLI command (seed 0):**

```bash
python scripts/train_diffusion.py \
    --dataset_path data/_repro/pusht_repro_seed0.npz \
    --run_dir      runs/_repro/diffusion_seed0 \
    --seed         0 \
    --steps        5000 \
    --device       cpu
```

**Outputs written to `--run_dir`:**

- `checkpoint.pt` — model weights and full architecture + schedule config.
- `config.json` — full training configuration.
- `train_summary.json` — final training loss.

---

## 4. Evaluation Protocol

**Scripts:** `scripts/eval_bc.py`, `scripts/eval_diffusion.py`

**Environment:** Same as dataset collection (`gym_pusht/PushT-v0`).

**Episode seeding:** Episode `i` uses reset seed `eval_seed + i`.

**Steps per episode:** Capped at `max_steps`. Episodes end earlier on termination or
truncation.

**Parameters used for `results/rq_exec_mode/`:**

| Parameter         | Value |
|-------------------|-------|
| `--episodes`      | 20    |
| `--max_steps`     | 200   |

**Success metric:** The environment's `terminated` flag is used as the primary success
signal. If the final `info` dict contains a `"success"` or `"is_success"` key, that
value takes precedence. No reward threshold is used as a success proxy.

In the `rq_exec_mode` run, `terminated` was never set within 200 steps for any method
or seed. `success_rate = 0.0` across all conditions. `return_mean` is the operative
metric for this comparison.

**BC evaluation:** At each step, the policy computes the mean action (no sampling).

**Diffusion evaluation — Open-loop chunk execution:**

1. On `policy.reset(episode_seed)`, the action cache is cleared.
2. On cache exhaustion, DDIM is called with `seed = episode_seed + chunk_index + sample_seed_base`.
3. Actions are returned from the cache in order (index 0 through H-1).

**Diffusion evaluation — Receding-horizon re-planning:**

1. On `policy.reset(episode_seed)`, the timestep counter is reset.
2. At every call to `policy.act(obs)`, DDIM is called with
   `seed = episode_seed + timestep_index + sample_seed_base`.
3. Only `seq[0, 0]` (first action) is returned; the rest is discarded.

**DDIM sampler configuration:**

| Parameter    | Value               |
|--------------|---------------------|
| `eta`        | 0.0 (deterministic) |
| `steps K`    | 10                  |
| Timesteps    | `linspace(0, T-1, K)` reversed |

**Exact CLI commands (seed 0):**

```bash
# BC evaluation
python scripts/eval_bc.py \
    --env_id          gym_pusht/PushT-v0 \
    --checkpoint_path runs/_repro/bc_seed0/checkpoint.pt \
    --out_path        results/rq_exec_mode/seed0/bc_eval.json \
    --seed            0 \
    --episodes        20 \
    --max_steps       200 \
    --device          cpu

# Diffusion evaluation (both modes)
python scripts/eval_diffusion.py \
    --env_id          gym_pusht/PushT-v0 \
    --checkpoint_path runs/_repro/diffusion_seed0/checkpoint.pt \
    --out_dir         runs/_repro/diffusion_seed0/eval_repro \
    --seed            0 \
    --episodes        20 \
    --max_steps       200 \
    --device          cpu
```

**Evaluation output JSON keys:**

`env_id`, `seed`, `episodes`, `episode_returns`, `episode_lengths`, `success_flags`,
`return_mean`, `return_std`, `success_rate`, `episode_len_mean`, `eval_seed_list_hash`.

---

## 5. Multi-seed Orchestration

The full pipeline for all seeds is run via `scripts/reproduce_multiseed.py`, which
handles dataset recording, BC training, BC evaluation, diffusion training, and
diffusion evaluation (both modes) per seed, skipping completed stages on re-run.

**Exact CLI command used for `results/rq_exec_mode/`:**

```bash
python scripts/reproduce_multiseed.py \
    --seeds            0 1 2 \
    --env_id           gym_pusht/PushT-v0 \
    --episodes_record  20 \
    --max_steps_record 200 \
    --steps_bc         3000 \
    --steps_diff       5000 \
    --episodes_eval    20 \
    --max_steps_eval   200 \
    --results_root     results/rq_exec_mode \
    --device           cpu
```

---

## 6. Multi-seed Aggregation

**Required file layout:**

```
results/rq_exec_mode/
    seed0/  bc_eval.json  diff_open_loop.json  diff_receding.json
    seed1/  ...
    seed2/  ...
```

**Validation:**

```bash
python scripts/validate_results.py \
    --seeds 0 1 2 \
    --results_root results/rq_exec_mode
```

Checks that all three JSON files exist per seed, are parseable, and contain
`success_rate`, `return_mean`, `return_std`, and `episode_len_mean`. Hard exits on any
missing file or key. Aggregation must not proceed on incomplete data.

**Aggregation:**

```bash
python scripts/aggregate_results.py \
    --seeds 0 1 2 \
    --results_root results/rq_exec_mode
```

Writes:
- `results/rq_exec_mode/per_seed.csv` — one row per (method, seed); all numeric fields.
- `results/rq_exec_mode/summary.csv` — one row per (method, metric); columns `method`,
  `metric`, `mean`, `std`, `n_seeds`. Cross-seed statistics use population std (ddof=0).

Methods reported: `bc`, `diff_open_loop`, `diff_receding`.

If any seed file is absent, `aggregate_results.py` exits with code 1. Partial
aggregation is not performed.

---

## 7. Experiment Logging

After each meaningful run, append an entry to `report/experiment_log.md` using the
template provided in that file. Each entry must record the commit hash, seed list,
config parameters, and all quantitative results. Entries are append-only; do not edit
prior entries.
