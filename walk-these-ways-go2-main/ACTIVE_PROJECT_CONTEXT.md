# Active Project Context

This file is the short entrypoint for the current Go2 gait-adaptation project.
Read this before editing training, evaluation, or visualization scripts.

Current source-of-truth plan:

```text
CURRENT_GAIT_ADAPTATION_PLAN.md
```

Documentation rule:

```text
Any change to the training/evaluation plan, any new validation result, and any
correction to the project interpretation must be recorded in the current project
documents immediately. Do not leave project-state changes only in chat.
```

The 2026-06-08 and 2026-06-11 handoff files are historical context. Do not use
their next-step sections without checking the current plan above.

## Goal

Build a high-level gait adaptation module on top of the frozen WTW Go2 low-level policy.
The high-level module should infer condition changes from proprioceptive history, select a gait,
and continuously tune gait parameters.

High-level output:
- discrete gait choice: pronking, trotting, bounding, pacing
- continuous residuals: frequency, duration, foot swing height, stance width, body pitch

The current goal is condition-aware gait-parameter adaptation. Keep the terrain-specific
reference metrics, but do not force gait differentiation by hard-tuning reward weights.
If one gait remains best under a condition, accept it and evaluate whether the learned
continuous parameters improve task metrics.

## Current Mainline

The active evaluation/training assets are under:

```text
logs/gait_condition_eval_v8_mainline
```

Important files:

```text
logs/gait_condition_eval_v8_mainline/template_eval_results.csv
logs/gait_condition_eval_v8_mainline/training_task_map/training_task_map_by_speed.csv
logs/gait_condition_eval_v8_mainline/gait_template_library/gait_template_library.csv
```

Do not use older `v1` to `v7` folders as defaults unless doing ablation or historical comparison.

## Active Training Scenes

Current training scenes:

1. `flat_trot_efficiency`
   - condition: `flat`
   - target gait: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

2. `ramp_up_trot_robustness`
   - condition: `ramp_up`
   - target gait: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

3. `rough_slope_trot_robustness`
   - condition: `rough_slope`
   - target gait: `trotting`
   - speeds: 0.5, 1.0, 1.5, 2.0

4. `push_lateral_pace_recovery`
   - condition: `push_lateral`
   - target gait: `pacing`
   - speed: 1.5

5. `stepping_stones_easy_bound_highspeed`
   - condition: `stepping_stones_easy`
   - target gait: `bounding`
   - speed: 2.0

Rejected or inactive:
- `rough_mid`: too difficult and overlaps with rough slope
- stairs: low-level WTW policy is not trusted for stairs
- low-friction/pronk: visual and score evidence did not support it
- walk: unstable in visual tests
- `push_hard/bounding`: evaluation-only; old mixed-direction frequent pushes were not clean evidence

## Runtime Defaults

Shared default paths and visualization constants live in:

```text
scripts/gait_project_config.py
```

Shared terrain and disturbance condition definitions live in:

```text
scripts/gait_conditions.py
```

Current `scripts/` mainline files are intentionally minimal:

```text
scripts/gait_project_config.py          shared paths and runtime defaults
scripts/gait_conditions.py              shared terrain/disturbance setup
scripts/train_high_level_ppo.py         reusable high-level PPO components
scripts/train_high_level_oracle_ppo.py  current oracle-condition sanity trainer
scripts/evaluate_fixed_gait_live_reward.py
                                         fixed-gait live reward audit
scripts/evaluate_gait_target_fairness.py
                                         fair gait target audit with per-gait
                                         continuous-parameter search
scripts/play_oracle_policy_training_map.py
                                         learned oracle policy visualization
scripts/visualize_oracle_training_results.py
                                         metrics plots and summary for oracle runs
scripts/test_oracle_policy_route.py     route-style multi-terrain policy test
scripts/evaluate_gait_templates.py      fixed-template evaluation helpers
scripts/play_task_gait_oracle.py        fixed-template single-scene playback
scripts/play_training_scenes_oracle.py  fixed-template active-scene playback
scripts/train.py                        original WTW low-level training entry
scripts/play.py                         original WTW low-level playback entry
```

Historical gait scans, old comparison players, and one-off analysis scripts were
removed from `scripts/` after the v8 mainline became the source of truth. Use the
CSV assets under `logs/gait_condition_eval_v8_mainline` instead of regenerating
old scan pipelines unless doing an explicit historical audit.

Visualization defaults:

```text
terrain_length = 30.0
terrain_width = 30.0
teleport_thresh = 3.0
edge_reset_margin = 3.0
```

Edge reset is enabled for visualization/evaluation when `edge_reset_margin` is passed.
It resets the robot near its own terrain patch boundary and does not count as a fall penalty.

## Common Commands

Visualize all active training scenes:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/play_training_scenes_oracle.py
```

Dry-run the scene commands:

```bash
python3 scripts/play_training_scenes_oracle.py --dry-run
```

## Next Step

The fixed-gait live reward audit has been completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845
```

Main readout:

- flat: live reward agrees with the target gait, trotting
- ramp/rough: trotting is near-tied with pronking, so selector signal is weak
- push: live reward prefers trotting; target pacing ranks last
- stones: live reward prefers pacing; target bounding ranks second

Important interpretation update:

```text
The target gait labels themselves are not yet fully validated.
```

The fixed-gait audit used the wrapper default gait templates with continuous
residuals fixed at zero. It is therefore evidence about the current live reward
under default continuous parameters, not a fair proof of each gait family's best
possible task performance.

Per-metric weighted gap decomposition has also been generated:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845/weighted_gap_decomposition.md
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845/weighted_gap_decomposition.csv
```

Do not continue reward-only selector training, curriculum training, or RMA/no-task
training as the next diagnostic. Do not implement the proposed reward-v5 weight
changes yet.

The immediate diagnostic is now a fair target-gait audit:

```text
For each task/speed/gait, compare the best achievable task score under an equal
continuous-parameter search budget, using a gait-agnostic objective.
```

This audit must search continuous gait parameters separately for every
task/speed/gait combination, not only evaluate `continuous residual = 0`.
It should report raw metrics, weighted scores, best parameters, and Pareto
trade-offs. Ambiguous tasks may produce a soft gait distribution instead of a
single hard target.

Recommended 4090 command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_gait_target_fairness.py \
  --full \
  --grid-mode action-space \
  --batch-size 32 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full \
  --skip-existing
```

Recommended `nohup` background command for SSH sessions:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
mkdir -p runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_gait_target_fairness.py \
  --full \
  --grid-mode action-space \
  --batch-size 96 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full \
  --skip-existing' \
  > runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full/nohup.log 2>&1 &
```

Monitor it with:

```bash
tail -f runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full/nohup.log
```

On a 24GB RTX 4090, start with `--batch-size 96 --repeats-per-config 2`
(`192` envs). If GPU memory still looks very low and simulation is stable, try
`--batch-size 128 --repeats-per-config 2` (`256` envs). If IsaacGym crashes or
CUDA OOMs, fall back to `--batch-size 32 --repeats-per-config 2`.

Troubleshooting:

```text
If the log says "argument --freq-residuals: expected one argument", the server
has an old `scripts/evaluate_gait_target_fairness.py`. Update the script. The
fixed version passes negative residual grids to child processes with
`--freq-residuals=-1.0,0.0,1.0` style syntax.

If the log says "Inplace update to inference tensor outside InferenceMode is not
allowed", the server has an old fair-audit script that used
`torch.inference_mode()` around env stepping. Update the script. The fixed
version uses `torch.no_grad()` for IsaacGym stepping.

If a child runs for a while and then exits with status 1 near the last batch,
update the script. Older versions recreated IsaacGym with a smaller env count
for the final partial batch; the fixed version pads the final batch and keeps a
constant sim/env count for the whole child process.

One observed old-log signature is:

```text
Finished batch 1/3
Finished batch 2/3
AttributeError: 'dict' object has no attribute 'command_curriculum'
```

This came from rebuilding the IsaacGym/WTW env for the partial final batch after
the global `Cfg` object had already been mutated. The fixed script avoids that
rebuild.
```

Output directory:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/<timestamp>
```

Completed result currently available:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full
```

Completeness check:

```text
fair_gait_grid_results.csv: 4536 data rows
14 task-speed rows * 4 gaits * 81 action-space parameter settings = 4536
best_by_task_speed.csv: 14 best rows plus header
best_by_task_speed_gait.csv: 56 best-per-gait rows plus header
```

Important coverage note:

```text
20260613_action_grid_full covers the active task-map training speed rows:
flat/ramp/rough at 0.5, 1.0, 1.5, 2.0;
push_lateral at 1.5;
stepping_stones_easy at 2.0.
```

In the trainer, because push/stones each have one active training-map speed, the
runtime sampled command ranges are expanded to:

```text
push_lateral: [1.2, 1.8]
stepping_stones_easy: [1.7, 2.0]
```

Therefore the completed fair audit should be treated as an active-row scan, not
as a full sampled-range scan for push/stones. To audit the actual sampled ranges,
run `scripts/evaluate_gait_target_fairness.py --training-range`. To also probe
extra diagnostic speeds such as push 0.5/1.0/2.0 and stones 1.0/1.5/2.0, run
with `--extended`.

Only after that audit should the project decide whether to accept the empirical
best gait, modify the performance reward, or build a score-derived soft gait
prior. See `CURRENT_GAIT_ADAPTATION_PLAN.md` for the current decision rules.

## Current Training Entry

The first sanity-check trainer is:

```text
scripts/train_high_level_oracle_ppo.py
```

This oracle trainer appends a task one-hot only as a sanity-check phase. Its default
training reward converts every online metric to a 0-1 score and computes a weighted
average. The weights are shared base weights plus condition-specific additions
derived from `reward_focus` in the task map. `--style-reward-scale` defaults to
`0.0`, so target gait labels are logged for analysis but are not used as hard
selector rewards unless explicitly enabled.

The shared base weights include progress, yaw tracking, orientation, lateral
drift, gait stability, action smoothness, action magnitude, action boundary
margin, and survival. The lateral-drift score combines lateral velocity with
body offset from the terrain patch centerline, and the high-level observation
history now includes signed and absolute lateral centerline offset. A
selector-hold mechanism and gait-switch penalty were added after v2 showed
frequent within-scene gait switching. Clearance is intentionally weak because it
is currently a foot-swing command proxy, not a direct terrain-relative scuffing
measurement.

It uses one IsaacGym sim with mixed envs. Each env is assigned one of the active
training tasks, including terrain type, push setting, target gait, style reward
strength, and continuous velocity sampling range.

This is an oracle-condition sanity check:

- policy input = proprioceptive history + task one-hot
- task one-hot is temporary and should be removed after verifying the reward/scene design
- output = categorical gait choice + continuous residual parameters
- PPO and actor-critic implementation are reused from `scripts/train_high_level_ppo.py`

Current v3 observation dimensions are:

```text
base high-level observation history: 510
task one-hot: 5
oracle observation: 515
```

Old v0-v2 checkpoints used 495-D oracle observations and are not compatible with
the v3 model architecture.

If oracle-condition training cannot separate the target gaits, do not train the
final proprioception-only version yet; fix reward/scene design first.

As of 2026-06-13, reward-only v4, selector-only long training, and single-task
selector-only probes have not proven stable target-gait selector separation.
The current unresolved question is whether the target gait labels themselves are
valid under a fair gait-agnostic evaluation with equal continuous-parameter
search per gait.

The first 100-iteration oracle run showed a reward-design warning: foot swing
height increased, but progress and action health slightly worsened. v2 added a
centerline reward, but metrics and route visualization still worsened because
the policy did not observe its lateral offset and gait selection switched too
frequently inside one scene. The next local run should be v3, validating
centerline-offset observations plus gait-stability reward/selector hold. Do not
start long training until this local validation is healthy.

After each local training run, visualize the latest checkpoint on the mixed
training map:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/play_oracle_policy_training_map.py \
  --run-dir runs/high_level_oracle_gait/<run_name> \
  --num-envs-per-task 4 \
  --steps 2000
```
