# Current Gait Adaptation Plan

Date: 2026-06-12

This is the current source-of-truth project note for the high-level Go2 gait
adaptation work. It supersedes the next-step recommendations in
`CONVERSATION_HANDOFF_20260608.md` and `CONVERSATION_HANDOFF_20260611.md`.

## Goal

Train a high-level policy on top of the frozen WTW Go2 low-level policy.

The high-level policy should:

- infer environment/terrain condition from proprioceptive history,
- select a gait family: pronking, trotting, bounding, pacing,
- tune continuous gait parameters: frequency, duration, footswing height,
  stance width, and body pitch,
- eventually remove oracle task labels and rely on RMA/proprioceptive evidence.

The active project goal is visible condition-driven gait behavior, not proving
that one gait is universally best. If live reward evidence says the current
training objective prefers a different gait from the task-map label, that is a
reward-design finding, not automatically a PPO failure.

## Active Code Path

Main training/evaluation files:

- `scripts/train_high_level_oracle_ppo.py`
  - mixed-task high-level PPO trainer,
  - optional task one-hot oracle observation,
  - optional RMA teacher/student latent,
  - optional selector-only training,
  - reward computed by the live `HighLevelGaitWrapper` metric table.

- `scripts/train_high_level_ppo.py`
  - reusable actor-critic, categorical gait head, Gaussian residual head,
  - selector-only policy/evaluation functions.

- `go2_gym/envs/wrappers/high_level_gait_wrapper.py`
  - 9D high-level action wrapper,
  - gait template and continuous parameter mapping,
  - live high-level reward and metric scores.

- `scripts/evaluate_high_level_policy_by_task.py`
  - deterministic independent checkpoint evaluation,
  - one task/speed at a time,
  - reports executed gait ratios and task metrics.

- `scripts/evaluate_fixed_gait_live_reward.py`
  - fixed-gait audit script,
  - one task/speed/gait at a time,
  - directly measures the current live training reward for fixed gait actions.

Main data assets:

- `logs/gait_condition_eval_v8_mainline/training_task_map/training_task_map_by_speed.csv`
- `logs/gait_condition_eval_v8_mainline/gait_template_library/gait_template_library.csv`
- `logs/gait_condition_eval_v8_mainline/template_eval_results.csv`

## Corrected Interpretation

### Target gait labels

`target_gait` in the task map is not automatically a training reward.

In the current trainer, target gait affects the reward only when:

```text
style_reward_scale > 0
```

With the default:

```text
--style-reward-scale 0.0
```

target gait labels are analysis labels, not hard or soft selector supervision.
This means reward-only v4 experiments should not be described as target-gait
supervised training.

### Raw reward comparisons

Do not compare raw `reward` or `weighted_metric_reward` across reward versions
as if they share the same scale.

Valid cross-run comparisons should focus on metrics such as:

- `vx_err`,
- `done_rate`,
- `action_clip_rate`,
- `gait_switch_rate`,
- `lateral_position_penalty`,
- `score_slip`,
- `score_clearance`,
- independent-eval gait ratios.

### Soft prior

Do not write a soft prior as hard-coded labels such as:

```text
flat/ramp/rough -> trot
push -> pace
stones -> bound
```

The prior should be derived from score evidence:

```text
score(task, speed, gait) -> softmax(score / temperature)
```

The next audit must first determine whether the live reward v4 ranking agrees
with the template-eval/task-map ranking.

## Verified Results So Far

### v1 unified reward

Result:

- reward nearly flat,
- `vx_err` not meaningfully improved,
- `action_clip_rate` worsened,
- target gait ratios only weakly changed.

Readout:

```text
v1 was not a healthy training objective.
```

### v2 centerline reward

Result:

- reward and `vx_err` worsened,
- lateral position penalty worsened,
- route test did not complete all segments.

Readout:

```text
v2 did not solve route stability.
```

### v3 centerline observation + gait switch handling

Result:

- `vx_err` improved versus v2,
- `action_clip_rate` dropped near zero,
- action smoothness improved,
- gait switching improved only slightly,
- gait differentiation remained unclear.

Readout:

```text
v3 improved engineering/action health, but did not prove gait differentiation.
```

### reward v4 + RMA + no-task

Run:

```text
runs/high_level_oracle_gait/20260610_rma_notask_reward_v4
```

Result:

- some performance metrics improved versus old baselines,
- independent eval showed trot-heavy behavior on flat/ramp/rough,
- push did not become pace-dominant,
- stones did not become bound-dominant.

Readout:

```text
RMA/no-task is not currently the main proven bottleneck, because reward-only
v4 itself has not been shown to create the desired selector ranking.
```

### reward v4 + RMA + task one-hot

Result:

- task one-hot did not create strong condition-specific gait separation.

Readout:

```text
The failure is not explained only by lack of task information or RMA.
```

### selector-only + task one-hot, 300 iterations

Result:

```text
mean target ratio stayed weak around checkpoints 50-200 and fell by 250-299.
```

Readout:

```text
Longer selector-only reward-v4 training did not amplify the weak target-gait trend.
```

### single-task selector-only probes

Result:

- flat target trot, but dominant gait was bound,
- ramp target trot, but distribution was nearly mixed,
- rough target trot, but dominant gait was bound,
- push target pace, but dominant gait was trot,
- stones target bound, but dominant gait was pace.

Readout:

```text
Mixed-task gradient competition is not the only issue. Single-task reward-only
training also does not reliably push the target gait.
```

## Current Main Doubt

The biggest unresolved question is:

```text
Does the current live reward v4 actually rank the target/template gait highest?
```

The offline template tables may say one gait is best, but the live PPO training
reward may rank a different gait higher because it uses a different metric set
and normalization.

Therefore, before adding a soft gait prior, run a fixed-gait live reward audit.

## Immediate Next Step

Run the quick live reward audit:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py \
  --num-envs 32 \
  --steps 1000 \
  --warmup-steps 50
```

For all active training speeds:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py \
  --full \
  --num-envs 32 \
  --steps 1000 \
  --warmup-steps 50
```

The output is written under:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/<timestamp>
```

Key columns:

- `weighted_metric_reward`,
- `score_progress`,
- `score_slip`,
- `score_orientation`,
- `score_clearance`,
- `vx_err_mean`,
- `done_rate`,
- `requested_gait_actual_ratio`.

## Decision Rules After Audit

### Case A: live reward agrees with target/template gait

Example:

```text
flat live best = trotting
push live best = pacing
stones live best = bounding
```

Then reward v4 contains the desired ranking, but PPO/selector optimization is
not extracting it. Next steps can include:

- lower selector entropy over time,
- stronger selector credit assignment,
- selector-only with immediate switching,
- soft prior as an optimization aid rather than reward correction.

### Case B: live reward disagrees with target/template gait

Example:

```text
flat live best = bounding
```

Then PPO may be correctly optimizing the current live reward. Next steps must be
chosen explicitly:

- accept the live reward optimum,
- change the performance reward so the intended gait is truly better,
- or add a soft prior derived from template/live score evidence.

### Case C: live reward rankings are close or noisy

Then the selector receives a weak/ambiguous reward signal. A soft distribution
prior is justified, but it should be based on measured score gaps and
temperature, not hard labels.

## Cleanup Rule

Old handoffs and earlier experiment summaries are historical evidence only.
They should not be used as the current next-step plan unless this file says so.

When adding a new experiment, its config/summary must explicitly state:

- whether target gait enters the reward,
- `style_reward_scale`,
- selector prior scale, if any,
- whether continuous residuals are trained,
- whether task one-hot is present,
- whether RMA is enabled,
- whether training speed is fixed or sampled over a range,
- whether metrics come from mixed training rollout or independent single-task eval.
