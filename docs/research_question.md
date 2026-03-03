# Research Question

## Motivation

Diffusion models have been applied to robot policy learning as a way to represent
multimodal action distributions over finite-horizon sequences. The key property
exploited in this setting is the model's ability to generate coherent action sequences
conditioned on the current observation, rather than predicting a single next action.
This approach was formalized in Chi et al. (2023) and has since appeared in several
manipulation and locomotion systems.

When a policy produces a sequence rather than a single action, the system must decide
how much of that sequence to execute before querying the policy again. Two limiting
strategies bracket the design space:

- **Open-loop chunk execution** treats the full predicted sequence as a plan to be
  followed without interruption. The policy is queried once per chunk, and the robot
  executes H actions in open loop.
- **Receding-horizon re-planning** discards all but the first action of the predicted
  sequence and queries the policy at every control timestep. This is the standard
  model-predictive control formulation applied to a learned generative policy.

Intermediate strategies — executing the first K < H actions before re-planning — form
a spectrum between these two extremes. This repository studies the two endpoints.

The choice of execution strategy has practical consequences. Open-loop execution
reduces the number of diffusion sampling calls per unit time, which reduces compute
load and latency. Receding-horizon re-planning allows the policy to correct for
prediction errors and environmental disturbances at every step, at the cost of
continuous inference. For diffusion policies specifically, where each sampling call
involves K denoising steps, the computational difference is significant.

Whether this compute tradeoff translates into a meaningful performance difference on
a manipulation task — and in which direction — is not obvious from first principles.

---

## Formal Statement

**Training setup.** A single diffusion policy and a single Gaussian MLP BC baseline
are trained on Push-T demonstrations under fixed hyperparameters. Training is not
repeated as part of the comparison; both execution strategies are evaluated from the
same trained checkpoint.

**Independent variable.** Execution strategy: open-loop chunk execution (H actions per
sample) vs. receding-horizon re-planning (1 action per sample, H actions predicted).

**Dependent variables.** Mean episodic return and mean episode length.

**Controlled variables.** Model architecture, diffusion schedule, sampler
configuration, training data, training seed, evaluation episode count, maximum episode
length, and observation normalization.

**Question.** Under identical training conditions, does execution strategy produce a
measurable and consistent difference in task return across multiple seeds?

---

## Hypothesis

No directional prediction is made. The null hypothesis is that execution strategy
produces no statistically meaningful difference in return within this experimental
setup. A non-null result in either direction would warrant further investigation with
larger datasets, longer horizons, and additional tasks.

The BC baseline is included as a reference point, not as a comparison target. It uses
single-step prediction by definition and does not participate in the execution-strategy
comparison.

---

## Empirical Outcome

Results from the three-seed experiment (`results/rq_exec_mode/`, commit 04defe5)
are summarized here at a high level.

**Diffusion vs BC.** Diffusion sequence modeling outperformed the Gaussian BC baseline
in `return_mean` consistently across all three seeds (cross-seed mean: diffusion ≈ 7.70
vs BC ≈ 3.98). This pattern was consistent in direction across seeds and grew in
magnitude with seed index. This result is reported as an empirical observation, not as
a claim about the general superiority of diffusion policies over BC; the dataset is
small and collected with a random policy.

**Open-loop vs receding-horizon.** In this PushT configuration, execution strategy did
not materially change return. The cross-seed difference in `return_mean` between the
two strategies was approximately +0.02 in favor of receding-horizon, which is well
within cross-seed noise. No consistent difference in episode length was observed. The
null hypothesis (no meaningful execution-strategy effect) is not rejected in this
experimental setup.

**success\_rate.** `success_rate = 0.0` for all methods and seeds. The Push-T
termination signal does not trigger within 200 steps under the evaluated policies.
`return_mean` is the primary interpretable metric for this run.

These results are scoped to a single environment with low-dimensional state, a small
random-action dataset, a fixed horizon of H = 8, and a fixed DDIM sampler with K = 10
steps. Generalizing these observations to other environments, demonstration qualities,
or horizon lengths requires additional experiments.

---

## What Is Held Constant

The following are identical across the two execution-strategy conditions:

- Trained model checkpoint (weights, architecture, schedule parameters).
- DDIM sampler configuration (eta = 0.0, K = 10 steps, linear schedule).
- Evaluation environment and task.
- Episode count (20) and maximum episode length (200).
- Observation normalization parameters.
- Master evaluation seed and per-episode reset seeds.
- Diffusion noise seed derivation logic (only the indexing variable differs: chunk
  index for open-loop, timestep index for receding-horizon).

---

## What Is Varied

The sole difference between the two conditions is:

- **Open-loop:** `policy.act(obs)` returns cached actions from a single sample;
  re-sampling occurs every H steps.
- **Receding-horizon:** `policy.act(obs)` calls the sampler at every step and returns
  the first action of the new sample.

No other code path, parameter, or configuration differs between conditions.

---

## Scope and Non-Goals

This study does not address:

- Whether the finding generalizes beyond Push-T.
- The effect of horizon length H on the relative performance of the two strategies.
- The effect of sampler step count K on receding-horizon latency.
- Optimal intermediate chunk sizes.
- The interaction between demonstration quality and execution strategy.

These are natural extensions but are outside the scope of this repository. See
`docs/threat_model.md` for a complete statement of limitations.

---

## References

Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023).
Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *Robotics: Science
and Systems (RSS)*.

Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. *ICLR
2021*.
