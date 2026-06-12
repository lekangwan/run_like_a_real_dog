# Conversation Handoff - Go2 High-Level Gait Adaptation

Date: 2026-06-08

> Historical note, superseded on 2026-06-12.
> Read `CURRENT_GAIT_ADAPTATION_PLAN.md` first. This file is useful for
> project history, but its training next steps are no longer the active plan.

This document is intended for migrating the current project to a new Codex
conversation. Read `ACTIVE_PROJECT_CONTEXT.md` first, then this file.

## User Profile And Working Style

- The user is an automation undergraduate preparing for graduate-school
  recommendation interviews, with about two months of preparation time.
- The project is based on Unitree Go2 and Walk-These-Ways Go2.
- The user is a beginner in reinforcement learning reward design and wants
  mechanism-level explanations, not just commands.
- Preferred interaction:
  - Work step by step.
  - Do not dump a full huge plan unless asked.
  - When the user questions a design, first judge whether the concern is
    reasonable, explain the reason, then decide whether to modify.
  - Do not blindly follow the user's command when it may be technically wrong.
  - Use actual repository evidence, actual paths, actual metrics, and actual
    runtime behavior.

## Repository And Current Source Of Truth

Project path:

```text
/home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
```

Source-of-truth entry file:

```text
ACTIVE_PROJECT_CONTEXT.md
```

Active mainline assets:

```text
logs/gait_condition_eval_v8_mainline
logs/gait_condition_eval_v8_mainline/template_eval_results.csv
logs/gait_condition_eval_v8_mainline/training_task_map/training_task_map_by_speed.csv
logs/gait_condition_eval_v8_mainline/gait_template_library/gait_template_library.csv
```

Do not default to older `v1`-`v7` folders unless doing historical comparison or
ablation.

Current `scripts/` has been cleaned to the active mainline:

```text
gait_project_config.py
gait_conditions.py
train_high_level_ppo.py
train_high_level_oracle_ppo.py
play_oracle_policy_training_map.py
visualize_oracle_training_results.py
test_oracle_policy_route.py
evaluate_gait_templates.py
play_task_gait_oracle.py
play_training_scenes_oracle.py
train.py
play.py
```

Historical scan builders, old gait-comparison players, plotting helpers, and
one-off analysis scripts were removed from `scripts/`. The v8 CSV assets are the
source of truth for template evidence and task-map data.

## Final Project Goal

Build a high-level gait adaptation module on top of a frozen WTW Go2 low-level
policy.

The high-level module should:

- infer condition changes from proprioceptive history,
- select one of several gait families,
- continuously tune gait parameters,
- eventually be tested in simulation and then on a real Unitree Go2.

Important: the current goal is not to forcibly create visual gait
differentiation by hard-tuning reward. If trot is genuinely best in many
conditions, accept it. The meaningful goal is condition-aware gait-parameter
adaptation and measurable improvement under different conditions.

## Current High-Level Policy Contract

Low-level WTW policy is frozen.

High-level action dimension: 9.

Action layout:

```text
0-3: discrete gait selector
     pronking, trotting, bounding, pacing

4-8: continuous residual gait parameters
     frequency
     duration
     foot swing height
     stance width
     body pitch
```

Current oracle training input:

```text
proprioceptive history + task one-hot
```

Current observation dimensions:

```text
base high-level observation history: 510
task one-hot: 5
oracle observation: 515
```

The task one-hot is temporary. It is used only for sanity checking that the
reward and scenes are learnable. The final version should remove the task label
and rely on proprioception/history.

Important compatibility note: v3 adds signed and absolute lateral centerline
offset to each high-level observation frame, so old v0-v2 checkpoints with
495-D oracle observations cannot be loaded into the new 515-D model.

## Active Training Scenes

There are five active training tasks:

1. `flat_trot_efficiency`
   - condition: `flat`
   - target gait label: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

2. `ramp_up_trot_robustness`
   - condition: `ramp_up`
   - target gait label: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

3. `rough_slope_trot_robustness`
   - condition: `rough_slope`
   - target gait label: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

4. `push_lateral_pace_recovery`
   - condition: `push_lateral`
   - target gait label: `pacing`
   - evidence speed: 1.5
   - training samples vx in roughly 1.2-1.8

5. `stepping_stones_easy_bound_highspeed`
   - condition: `stepping_stones_easy`
   - target gait label: `bounding`
   - evidence speed: 2.0
   - training samples vx in roughly 1.7-2.0

Important interpretation: target gait labels are currently analysis references
and optional style-shaping references. With default `style_reward_scale=0.0`,
they are not hard selector rewards.

## Rejected Or Inactive Ideas

- `rough_mid`: removed because it was too difficult and overlaps with
  `rough_slope`.
- stairs: excluded because the current WTW low-level policy is not trusted for
  stairs.
- low-friction/pronk: visual inspection and score evidence did not support it.
- walk gait: tested visually; unstable and worse than trot, so dropped.
- push_hard/bounding: old mixed-direction, high-frequency pushes produced bound
  advantage, but this was not clean evidence; kept only as evaluation/ablation.
- pure speed-only gait selector: rejected as insufficient, because speed alone
  tends to collapse to the globally best gait.
- hard reward tuning to force different gaits: rejected as scientifically weak
  and risky for real-machine transfer.
- direct continuous-parameter-only demo without gait selection: rejected because
  visual differentiation was too weak for demonstration value.

## Current Reward Design

Training reward is a unified normalized metric table.

Each online metric is converted to an approximately 0-1 score, then a weighted
average is computed. This avoids directly adding raw quantities with very
different scales.

Current metric names in `HighLevelGaitWrapper.TASK_REWARD_NAMES`:

```text
progress
yaw_tracking
orientation
pitch_rate
roll_rate
yaw_rate
lateral_drift
vertical_bounce
slip
energy
clearance
gait_stability
action_smoothness
action_magnitude
action_boundary_margin
survival
```

Shared base weights in `scripts/train_high_level_oracle_ppo.py`:

```python
BASE_METRIC_WEIGHTS = {
    "progress": 1.0,
    "yaw_tracking": 0.3,
    "orientation": 0.3,
    "lateral_drift": 0.8,
    "gait_stability": 0.4,
    "action_smoothness": 0.7,
    "action_magnitude": 0.6,
    "action_boundary_margin": 0.8,
    "survival": 2.0,
}
```

Condition-specific additions are read from `reward_focus` in the task map:

```python
"progress": {"progress": 1.0}
"recovery_progress": {"progress": 1.0}
"low_energy": {"energy": 0.6}
"low_slip": {"slip": 0.8}
"low_vertical_bounce": {"vertical_bounce": 0.6}
"low_lateral_drift": {"lateral_drift": 0.8}
"orientation_stability": {"orientation": 0.9}
"pitch_control": {"pitch_rate": 0.8, "orientation": 0.4}
"low_roll_pitch_rate": {"roll_rate": 0.6, "pitch_rate": 0.6}
"low_roll_rate": {"roll_rate": 0.8}
"low_yaw_rate": {"yaw_rate": 0.8}
"low_done_rate": {"survival": 1.0}
"foot_clearance": {"clearance": 0.35}
"low_scuffing": {"clearance": 0.15}
```

Important caveat: current `clearance` is still a foot-swing command proxy, not a
true terrain-relative foot scuffing measurement. This is why its weight was
reduced after the first 100-iteration run.

After v1 visualization, the robot showed severe lateral offset and could fall
off the side before reaching the second route segment. v2 added a centerline
penalty and raised lateral/action-health weights, but v2 still worsened several
training metrics and showed frequent within-scene gait switching. The current
v3 fix is mechanism-level rather than only weight-level:

- add signed and absolute lateral centerline offset into high-level observations,
- add `gait_stability` and `gait_switch_penalty`,
- apply a short selector hold so the executed gait cannot flip every high-level
  step,
- log gait ratios, clip rate, and switch rate from the actually executed
  smoothed/held high-level action.

The final reward is:

```text
weighted_metric_reward - style_reward_scale * selector_reference_penalty
```

Default:

```text
style_reward_scale = 0.0
```

So reward normally equals `weighted_metric_reward`.

## Current Training Implementation

Training script:

```text
scripts/train_high_level_oracle_ppo.py
```

It uses:

- IsaacGym mixed envs.
- One PPO actor-critic policy.
- Each env is assigned one training task.
- All task envs are trained together in one rollout.
- PPO batches are mixed globally across tasks.
- On reset, velocity is resampled within that task's vx range.

It does not train one terrain at a time. It is not staged curriculum currently.

Default recent training parameters:

```text
num_envs = 256
num_steps = 32
iterations = 100 for local sanity check
value_coef = 0.5
entropy_coef = 0.003
lr = 3e-4
save_interval = 50
```

## Current Progress

The first oracle run completed:

```text
runs/high_level_oracle_gait/20260607_local_env256_iter100_unifiedreward_v0
```

Observed result:

- Training completed and metrics were stable.
- Reward barely improved.
- `reward` early10 about 0.6608, late10 about 0.6610.
- `vx_err` worsened from about 0.495 to about 0.525.
- `score_progress` worsened.
- `score_clearance` rose strongly from about 0.702 to about 0.810.
- `footswing_height_mean` rose from about 0.093 to about 0.103.
- `action_clip_rate` rose from about 0.102 to about 0.187.

Interpretation:

The policy was exploiting the clearance proxy by raising foot swing height,
while progress and action health worsened. This run proved that the pipeline
works, but the reward needed correction before long training.

Changes made after v0:

- Added `action_boundary_margin`.
- Reduced clearance weights.
- Increased action smoothness/action magnitude/boundary weights.
- Lowered default entropy coefficient to 0.003.
- Added per-task continuous parameter and clip-rate logging.
- Added mixed-map visualization for latest oracle checkpoint.

These changes are not yet validated by a new 100-iteration run.

## Visualization Workflow

New script:

```text
scripts/play_oracle_policy_training_map.py
```

Purpose:

- Load a trained oracle high-level checkpoint.
- Create a mixed env containing all active training conditions.
- Run the latest learned high-level policy, not the fixed gait-template oracle.
- Show per-env gait and continuous parameter choices in the terminal.

This was tested with a 1-step no-render rollout using the old v0 checkpoint and
successfully loaded:

```text
runs/high_level_oracle_gait/20260607_local_env256_iter100_unifiedreward_v0/checkpoints/high_level_000099.pt
```

Visualization command after each local run:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

CUDA_VISIBLE_DEVICES=0 python3 scripts/play_oracle_policy_training_map.py \
  --run-dir runs/high_level_oracle_gait/<run_name> \
  --num-envs-per-task 4 \
  --steps 2000 \
  --print-interval 200
```

Use `--no-render` only for quick script checks.

## Next Immediate Step

Run a new local 100-iteration sanity check with centerline observations and gait
stability:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

CUDA_VISIBLE_DEVICES=0 python3 scripts/train_high_level_oracle_ppo.py \
  --run-name 20260608_local_env256_iter100_unifiedreward_v3_centerline_obs_switch \
  --num-envs 256 \
  --num-steps 32 \
  --iterations 100 \
  --save-interval 50 \
  --value-coef 0.5 \
  --entropy-coef 0.003 \
  --lr 3e-4
```

After it finishes:

1. Inspect `metrics.csv`.
2. Compare early10 vs late10:
   - `reward`
   - `weighted_metric_reward`
   - `vx_err`
   - `lateral_position_penalty`
   - `gait_switch_rate`
   - `gait_switch_penalty`
   - `score_progress`
   - `score_clearance`
   - `score_gait_stability`
   - `score_action_smoothness`
   - `score_action_magnitude`
   - `score_action_boundary_margin`
   - `action_clip_rate`
   - per-task `*_footswing_height_mean`
   - per-task `*_action_clip_rate`
   - per-task `*_gait_switch_rate`
3. Run visualization:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/play_oracle_policy_training_map.py \
  --run-dir runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v3_centerline_obs_switch \
  --num-envs-per-task 4 \
  --steps 2000 \
  --print-interval 200
```

Then run the route-style test:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/test_oracle_policy_route.py \
  --run-dir runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v3_centerline_obs_switch \
  --segment-length 8.0 \
  --max-steps 5000 \
  --print-interval 100
```

Then generate plots and summary:

```bash
python3 scripts/visualize_oracle_training_results.py \
  --run-dir runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v3_centerline_obs_switch \
  --baseline-run-dir runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v2_centerline
```

Do not run 1000 iterations or server overnight training until v3 shows healthier
centerline tracking, action health, and gait-switch stability than v2.

## Environment Notes

Local environment:

```text
conda env: go2_wtw
Python: 3.8
IsaacGym: /home/lekangwan/isaacgym
GPU: local RTX 4060 8GB
```

Server environment was also configured earlier, but local 4060 is enough for
short 256-env sanity checks. Server should be reserved for long overnight runs
after local validation.

Important runtime quirks:

- If `isaacgym` cannot be imported, ensure `PYTHONPATH`/editable install points
  to `/home/lekangwan/isaacgym/python` and the conda env is active.
- IsaacGym may need to write to `~/.cache/torch_extensions`.
- Running without `conda activate go2_wtw` may fail to find `ninja`.
- Shared server jobs should use `CUDA_VISIBLE_DEVICES=<id>`, `nohup`, and careful
  `nvidia-smi` checks.
- Previous OOM/leak-like behavior was caused by graph retention / missing
  inference guards and/or too many envs; current PPO loop uses inference mode,
  detached buffers, `gc.collect`, and `torch.cuda.empty_cache`.

## Important Pitfalls Already Encountered

1. IsaacGym import issue:
   - Official demo worked, project failed because the project env did not have
     `isaacgym` installed/importable.

2. Missing pretrained run:
   - `scripts/play.py` failed with `IndexError` because expected `runs/` assets
     were missing.

3. CUDA device-side assert:
   - Occurred during early gait scan with invalid env/config sizing.

4. CUDA OOM:
   - Earlier training retained computation graphs or accumulated tensors.
   - Fixed by detaching rollout tensors and using inference mode around env
     stepping.

5. Terrain device mismatch:
   - `torch.meshgrid` / env origin tensors mixed CPU and CUDA in terrain origin
     generation; fixed earlier in terrain/env code.

6. Terrain boundary/cliff issue:
   - Robots could fall off terrain patches.
   - Edge-reset logic was added so boundary resets do not count as fall penalty.

7. Wrong terrain previews:
   - User requested actual IsaacGym visualization before long evaluation, not
     hand-drawn preview images.

8. Stepping-stones terrain initially too difficult:
   - First version had narrow high pillars and deep gaps.
   - It was adjusted to `stepping_stones_easy`.

9. Reward dominance:
   - Composite scores often collapsed to trot.
   - Hard tuning to force gait differentiation was judged weak.
   - Current approach keeps task metrics but does not force different gaits.

10. Clearance proxy exploit:
   - v0 oracle training raised foot swing height to improve reward while
     worsening progress/action health.
   - Clearance weights were reduced and action boundary margin added.

## Things A New Conversation Must Not Forget

- Read `ACTIVE_PROJECT_CONTEXT.md` first.
- Then read this file.
- Do not use old v1-v7 folders as active defaults.
- `scripts/` has been cleaned to the active mainline; do not reference removed
  old scan/comparison scripts unless explicitly restoring historical tooling.
- Do not assume target gait labels are hard rewards; default style scale is 0.
- Do not start long training just because a short run completes.
- The next job is v3 local 100-iteration validation with centerline-offset
  observations and gait-switch stability.
- The success criterion is healthier metrics and visual behavior, not just a
  slightly higher scalar reward.
