# Threat Model and Scope Limitations

This document explicitly states what this repository does not address. Its purpose is
to prevent overinterpretation of the results and to communicate scope to readers
considering building on this work.

---

## What This Repository Is

A controlled, single-task simulation study examining the effect of execution strategy
on the performance of a diffusion-based action-sequence policy trained on random
demonstrations. The environment is Push-T with low-dimensional state. All experiments
are run in simulation on a single CPU. Reproducibility and determinism are the
primary engineering concerns.

---

## Success Rate Metric

**`success_rate` is not used as the primary metric in this study.** In the current
evaluator (`src/diffusion_policy_manipulation/eval/rollout_evaluator.py`), success is
derived from the `terminated` flag returned by the environment step function. If the
`info` dict provides an explicit `"success"` or `"is_success"` key, that takes
precedence; otherwise, `terminated` is used as a binary proxy.

In the `rq_exec_mode` experiment, `terminated` was never set within 200 steps for any
method or seed. The Push-T environment in gym-pusht 0.1.6 does not raise a `terminated`
signal when the block is placed correctly within the allotted steps under the evaluated
policies. As a result, `success_rate = 0.0` for all methods and all seeds. This is a
property of the interaction between the current evaluator logic and this specific
environment version, not evidence that the policies are uniformly unsuccessful.

`return_mean` is the primary interpretable metric for all comparisons in this study.
`success_rate` is reported for completeness but should not be used to rank methods.

---

## What This Repository Is Not

### 1. A sim-to-real transfer study

No physical hardware is involved. The Push-T environment has no correspondence to any
real robotic system. Latency, actuator dynamics, communication delays, and sensor
noise are absent from the simulation. Findings about execution strategy in simulation
may not hold on a physical robot where re-planning latency is non-trivial.

### 2. A latency benchmarking study

Policy inference time is recorded as `mean_policy_time` if populated, but is not the
focus of any experiment. Measurements are taken on CPU with no optimization. GPU
inference, batched inference, and real-time constraints are not evaluated. Claims about
the practical feasibility of receding-horizon re-planning at robot control frequencies
cannot be made from this data.

### 3. A multi-task generalization study

All training and evaluation uses a single task with a single demonstration distribution.
No claim is made about the generality of the execution-strategy finding to other tasks,
observation modalities, or action spaces.

### 4. A domain randomization or robustness study

The environment uses fixed physics parameters. No noise is injected into observations
or actions. No perturbations are applied during evaluation. The policy is evaluated
under conditions identical to its training distribution. Robustness to distribution
shift is not addressed.

### 5. A safety study

No safety constraints, collision avoidance, or constraint satisfaction is modeled.
Results should not be interpreted as evidence for or against the safety of either
execution strategy on a real system.

### 6. A demonstration-quality study

Demonstrations are collected with a uniform random policy. The dataset does not reflect
expert behavior, teleoperation data, or motion-planned trajectories. The effect of
demonstration quality on the relative performance of execution strategies is not
studied.

### 7. A hyperparameter sensitivity study

Horizon H, diffusion steps T, sampler steps K, beta schedule, batch size, and
architecture width are fixed. No ablations are performed over these parameters. The
results reflect a single configuration chosen for computational tractability on CPU.

### 8. A benchmark comparison

No comparison is made against published methods. Returns are reported in absolute terms
within this experimental setup only. Numbers reported here are not directly comparable
to numbers in other papers that use different environments, datasets, or evaluation
protocols — including the original Diffusion Policy paper.

---

## Known Confounds

**Random-action demonstrations.** The training distribution is narrow and non-expert.
A policy that achieves positive return on this dataset may behave differently when
trained on expert demonstrations. The execution-strategy comparison is valid within
this setup but may change direction or magnitude with higher-quality data.

**Single-seed dataset per experiment run.** The dataset for each seed is recorded once
from that seed's random policy. Different seeds produce different trajectory coverage.
Cross-seed variation in dataset content contributes to cross-seed return variance and
is not separately controlled.

**Push-T task structure.** Push-T is a planar pushing task with a block and a target
region. It is relatively short-horizon and low-dimensional. Execution strategies that
produce equivalent results on this task may behave differently on tasks with longer
horizons, higher state-space dimensionality, or more contact-rich dynamics.

**episode\_len\_mean = 200.0 for all methods.** All episodes run to the maximum step
cap. This means episode length is not an informative signal in the current setup; it
simply reflects that no method terminates an episode before the cap.

---

## Summary

This repository should be read as a reproducible baseline and a methodology reference
for controlled execution-strategy experiments, not as a general claim about the
superiority of either open-loop or receding-horizon diffusion policy execution. Any
extension of these findings beyond the stated experimental setup requires additional
experimentation.
