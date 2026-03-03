# Experiment Log

This log is updated incrementally as experiments are run.

## Conventions
- Run IDs: timestamp + short slug
- Always record: commit hash, seed(s), config path, dataset version, eval protocol version, key metrics

---

## Entry template — copy and fill in after each meaningful run

```
### [YYYY-MM-DD] <short slug>

Date:         YYYY-MM-DD
Commit:       <git short hash>
Config:       configs/<config_file>.json  (or inline)
Dataset:      data/<dataset>.npz  (seed=N, episodes=E, max_steps=S)
Seeds:        [0, 1, 2]
Execution mode(s):  open_loop | receding_horizon
Horizon H:    8
Sampler steps K:  10

#### Quantitative results

| Method            | Exec mode    | success_rate | return_mean | return_std | episode_len_mean |
|-------------------|--------------|--------------|-------------|------------|-----------------|
| Gaussian BC       | —            | 0.000        | 0.0000      | 0.0000     | 0.00            |
| Diffusion         | open_loop    | 0.000        | 0.0000      | 0.0000     | 0.00            |
| Diffusion         | receding     | 0.000        | 0.0000      | 0.0000     | 0.00            |

Δ success_rate  (receding − open_loop):  +0.000
Δ return_mean   (receding − open_loop):  +0.0000
Δ episode_len_mean (receding − open_loop): +0.00

#### Observations
- Stability:          [describe control smoothness differences]
- Failure patterns:   [describe failure modes]
- Latency:            [mean_policy_time open_loop vs receding]

#### Limitations
- Single environment (Push-T low-dim state)
- Single horizon H=8
- Fixed sampler steps K=10
- No image observations
```

---

### [2026-03-03] rq-exec-mode-3seed

Date:     2026-03-03
Commit:   04defe5
Dataset:  per-seed random-policy collection (seed=N, episodes=20, max_steps=200)
Seeds:    [0, 1, 2]
Results:  `results/rq_exec_mode/per_seed.csv`, `results/rq_exec_mode/summary.csv`

#### Configuration

| Parameter             | Value                                          |
|-----------------------|------------------------------------------------|
| Environment           | gym_pusht/PushT-v0                             |
| Observation dim       | 5                                              |
| Action dim            | 2                                              |
| Dataset episodes      | 20 per seed                                    |
| Dataset max\_steps    | 200                                            |
| Dataset policy        | Uniform random                                 |
| BC training steps     | 3 000                                          |
| Diffusion train steps | 5 000                                          |
| Horizon H             | 8                                              |
| Diffusion T           | 50                                             |
| beta\_start           | 1 × 10⁻⁴                                      |
| beta\_end             | 0.02                                           |
| DDIM steps K          | 10                                             |
| DDIM eta              | 0.0 (deterministic)                            |
| Eval episodes         | 20 per seed                                    |
| Eval max\_steps       | 200                                            |
| Device                | CPU                                            |

#### Quantitative results — per-seed

| Method            | Exec mode | Seed | return\_mean | return\_std | episode\_len\_mean |
|-------------------|-----------|-----:|-------------:|------------:|-------------------:|
| Gaussian BC       | —         | 0    |         1.97 |        6.55 |              200.0 |
| Gaussian BC       | —         | 1    |         4.09 |       12.82 |              200.0 |
| Gaussian BC       | —         | 2    |         5.88 |       13.34 |              200.0 |
| Diffusion         | open\_loop | 0   |         5.65 |       13.25 |              200.0 |
| Diffusion         | open\_loop | 1   |         7.05 |       16.93 |              200.0 |
| Diffusion         | open\_loop | 2   |        10.39 |       21.27 |              200.0 |
| Diffusion         | receding  | 0    |         5.68 |       13.31 |              200.0 |
| Diffusion         | receding  | 1    |         7.07 |       16.98 |              200.0 |
| Diffusion         | receding  | 2    |        10.41 |       21.28 |              200.0 |

#### Quantitative results — cross-seed aggregated (mean ± std, n=3)

| Method              | Exec mode  | return\_mean       | return\_std        | success\_rate |
|---------------------|------------|--------------------|--------------------|---------------|
| Gaussian BC         | —          | **3.98 ± 1.60**    | 10.90 ± 3.08       | 0.00          |
| Diffusion           | open\_loop | **7.70 ± 1.99**    | 17.15 ± 3.28       | 0.00          |
| Diffusion           | receding   | **7.72 ± 1.98**    | 17.19 ± 3.26       | 0.00          |

Δ return\_mean (receding − open\_loop):    +0.02 (negligible)
Δ return\_std  (receding − open\_loop):    +0.04 (negligible)
Δ episode\_len\_mean (receding − open\_loop): 0.00

#### Observations

- **Diffusion vs BC.** Diffusion sequence modeling outperformed Gaussian BC across all
  three seeds. Cross-seed mean return: BC = 3.98, diffusion ≈ 7.70. The gap is consistent
  in direction across seeds and grows with seed index (seed 2 shows the largest absolute
  gap: BC 5.88 vs diffusion ~10.40).

- **Open-loop vs receding-horizon.** The difference between the two execution strategies
  is negligible at this scale. The cross-seed Δ return\_mean is +0.02 in favor of
  receding-horizon, well within cross-seed noise. No stability or control-smoothness
  difference was observable from the episode return distributions.

- **success\_rate = 0.0 for all methods.** The Push-T termination condition does not
  trigger within 200 steps under the evaluated policies. The `terminated` flag is used
  as the success proxy in our evaluator, and it was never set. `return_mean` is the
  primary interpretable metric for this run.

- **episode\_len\_mean = 200.0.** All episodes ran to the max\_steps cap for all methods
  and seeds. This is consistent with success\_rate = 0.0 and with policies that do not
  solve the task within the evaluation horizon.

- **Latency.** `mean_policy_time` was not populated in this run (CPU, no profiling).

#### Limitations

- Single environment (Push-T, 5-dim state, random-action demonstrations).
- Small dataset regime (20 episodes per seed).
- Single horizon H = 8; interaction between H and execution strategy not studied.
- Fixed DDIM sampler (K = 10, eta = 0.0); sampler ablations not performed.
- success\_rate uninformative due to evaluator termination-signal fallback.
- No image observations; no transformer backbone.

---
