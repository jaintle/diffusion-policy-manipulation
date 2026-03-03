# Diffusion Policy Manipulation

A controlled study of diffusion-based action-sequence policies for robotic manipulation.
The repository trains an MLP denoiser on Push-T demonstrations and evaluates how
**execution strategy** — open-loop chunk execution vs. receding-horizon re-planning —
affects task performance under identical training conditions.
All experiments are deterministic, multi-seed, and locally reproducible.

---

## Research Question

Given a diffusion policy trained to predict action sequences, does execution strategy
affect manipulation success rate and episode length in a controlled single-task setting?

Two strategies are compared under identical training:

- **Open-loop chunk execution** — sample a full H-step sequence; execute all H actions
  before re-planning.
- **Receding-horizon re-planning** — sample a full H-step sequence; execute only the
  first action; re-plan at every timestep.

Training data, model architecture, diffusion schedule, and sampler configuration are
held constant. Execution strategy is the sole independent variable.

This is a controlled evaluation study. We do not claim state-of-the-art performance.
We do not compare against prior methods. We isolate a single execution-time decision
and measure its effect on a standard benchmark task.

---

## Repository Structure

```
src/diffusion_policy_manipulation/   core library
    data/        dataset loading and observation/action normalization
    envs/        Push-T environment wrapper with deterministic seeding
    models/      MLP BC policy, MLP denoiser, diffusion schedule, DDIM sampler
    train/       BC and diffusion training loops
    eval/        rollout evaluator, execution-strategy policy wrappers, latency

scripts/
    record_dataset.py            collect Push-T demonstrations
    train_bc.py                  train Gaussian MLP BC baseline
    eval_bc.py                   evaluate BC checkpoint
    train_diffusion.py           train MLP diffusion denoiser
    eval_diffusion.py            evaluate diffusion checkpoint (both exec modes)
    reproduce_multiseed.py       full multi-seed pipeline orchestrator
    validate_results.py          verify per-seed JSON outputs exist and are complete
    aggregate_results.py         compute mean +/- std; write summary.csv, per_seed.csv
    smoke_determinism.py         Phase 1 smoke test
    smoke_dataset.py             Phase 2 smoke test
    smoke_bc.py                  Phase 3 smoke test
    smoke_diffusion_sampler.py   Phase 4 smoke test
    smoke_execution_modes.py     Phase 5 smoke test
    smoke_diffusion_train_eval.py Phase 6 smoke test
    smoke_multiseed.py           Phase 7 smoke test

configs/                         per-experiment YAML/JSON configs (not yet populated)
data/                            datasets (gitignored)
runs/                            training checkpoints and logs (gitignored)
results/                         per-seed JSON outputs and aggregated CSVs (gitignored)
report/
    experiment_log.md            incremental log of meaningful experimental runs
docs/
    protocol.md                  exact dataset, training, and evaluation protocol
    research_question.md         academic framing of the research question
    threat_model.md              explicit scope limitations and non-goals
tests/                           unit tests
```

---

## Determinism Guarantees

Reproducing results on the same hardware produces byte-identical JSON outputs.

**Global seed control.** Python `random`, NumPy, and PyTorch CPU and CUDA seeds are
set from a single integer at the start of every script. `torch.backends.cudnn.deterministic`
is enabled and `torch.use_deterministic_algorithms(True)` is applied where supported.

**Dataset generation.** Episode resets and action sampling use separate, fixed
`np.random.RandomState` objects seeded from the master seed. Recording the same seed
twice produces an identical `.npz` file.

**Training.** Each training step derives a fresh `RandomState(seed + step)` for batch
sampling and a fresh `torch.Generator(seed + step + 777)` for diffusion noise and
timestep draws. No persistent global RNG state is consumed between steps.

**Diffusion sampling.** DDIM is used with `eta = 0.0` (deterministic). A
`torch.Generator` is created fresh per sampling call, seeded from a deterministic
function of the episode seed and step index. No state leaks between calls.

**Evaluation.** Episode `i` resets with seed `master_seed + i`. Execution-strategy
wrappers derive sampling seeds from `episode_seed + chunk_index` (open-loop) or
`episode_seed + timestep_index` (receding-horizon). The set of reset seeds is hashed
and recorded in every output JSON for verification.

**Outputs.** JSON files contain all scalar metrics and per-episode arrays. Given
identical inputs, two runs produce byte-identical JSON.

---

## Installation

Python >= 3.10 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

The Push-T environment (`gym-pusht`) requires `pymunk >= 7.0`. On `pymunk 7.2`, a
one-line patch to the gym-pusht source is needed to replace the removed
`Space.add_collision_handler` API:

```python
# gym_pusht/envs/pusht.py  — replace:
self.collision_handeler = self.space.add_collision_handler(0, 0)
self.collision_handeler.post_solve = self._handle_collision
# with:
self.space.on_collision(0, 0, post_solve=self._handle_collision)
self.collision_handeler = None
```

---

## Quick Smoke Tests

Each smoke test runs in under two minutes on CPU. Run them in order after installation.

```bash
# Phase 1 — deterministic seeding and environment creation
python scripts/smoke_determinism.py

# Phase 2 — dataset recording and loading
python scripts/smoke_dataset.py

# Phase 3 — Gaussian MLP BC training and evaluation
python scripts/smoke_bc.py

# Phase 4 — diffusion schedule, denoiser, and DDIM sampler
python scripts/smoke_diffusion_sampler.py

# Phase 5 — open-loop and receding-horizon execution wrappers
python scripts/smoke_execution_modes.py

# Phase 6 — diffusion training and evaluation
python scripts/smoke_diffusion_train_eval.py

# Phase 7 — multi-seed pipeline, validation, and aggregation
python scripts/smoke_multiseed.py
```

All smoke tests exit with code 0 on success and print a PASSED line to stdout.

---

## Reproducing Main Results

The full multi-seed experiment follows five steps. All commands assume the virtual
environment is active and the working directory is the repository root.

**Step 1. Generate the dataset.**

```bash
python scripts/record_dataset.py \
    --env_id gym_pusht/PushT-v0 \
    --seed 0 --episodes 200 --max_steps 300 \
    --out data/pusht_main.npz
```

**Step 2. Train the Gaussian BC baseline (per seed).**

```bash
python scripts/reproduce_multiseed.py \
    --seeds 0 1 2 \
    --episodes_record 200 --max_steps_record 300 \
    --steps_bc 5000 --steps_diff 5000 \
    --episodes_eval 50 --max_steps_eval 300 \
    --results_root results/main \
    --device cpu
```

`reproduce_multiseed.py` orchestrates the full pipeline for each seed: dataset
recording, BC training, BC evaluation, diffusion training, and diffusion evaluation
for both execution modes. Completed stages are skipped on re-run.

**Step 3. Validate outputs.**

```bash
python scripts/validate_results.py \
    --seeds 0 1 2 \
    --results_root results/main
```

**Step 4. Aggregate across seeds.**

```bash
python scripts/aggregate_results.py \
    --seeds 0 1 2 \
    --results_root results/main
```

This writes `results/main/per_seed.csv` and `results/main/summary.csv`.

---

## Results

After aggregation, `results/main/summary.csv` contains one row per method-metric pair:

| method            | metric           | mean   | std    | n_seeds |
|-------------------|------------------|--------|--------|---------|
| bc                | success_rate     | —      | —      | 3       |
| diff_open_loop    | success_rate     | —      | —      | 3       |
| diff_receding     | success_rate     | —      | —      | 3       |
| ...               | ...              | —      | —      | ...     |

Numeric results are populated after running `reproduce_multiseed.py` and
`aggregate_results.py`. See `report/experiment_log.md` for entries recorded after
each meaningful run.

---

## Limitations

This repository is deliberately scoped. Known limitations:

- **Single environment.** All experiments use Push-T with low-dimensional state. Results
  may not transfer to other tasks, observation modalities, or action spaces.
- **Small dataset regime.** Demonstrations are collected with random actions. Coverage
  of the state space is limited.
- **Fixed horizon.** Sequence length H is held constant. The interaction between horizon
  length and execution strategy is not studied.
- **Fixed sampler.** DDIM with a fixed step count K is used throughout. Sampler choice
  and step count are not varied.
- **No vision.** Observations are low-dimensional state vectors. Image-conditioned
  diffusion is not implemented.
- **No transformer backbone.** The denoiser is an MLP. Attention-based architectures
  are not evaluated.
- **Simulation only.** No sim-to-real transfer is attempted. Latency and safety
  constraints of physical hardware are not modeled.

---

## Future Work

Extensions that are in scope for follow-on work, listed without priority:

- Larger demonstration datasets with task-conditioned data collection.
- Vision-conditioned denoiser with a frozen encoder.
- Latency profiling on accelerated hardware to measure the real cost of re-planning.
- Additional execution strategies (e.g., partial chunk execution).
- Transfer to a second environment to test generalization of the execution-strategy
  finding.

---

## Citation

If this codebase is useful to your work, please cite it as:

```
@misc{diffusion-policy-manipulation-2026,
  author = {Jain, Abhinav},
  title  = {Diffusion Policy Manipulation: Execution Strategy Study},
  year   = {2026},
  url    = {https://github.com/imabhi80/diffusion-policy-manipulation}
}
```
