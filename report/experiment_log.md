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
