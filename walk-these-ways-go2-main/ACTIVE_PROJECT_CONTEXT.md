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
The high-level module should infer condition changes from proprioceptive history and tune
locomotion behavior through both a discrete gait selector and continuous gait parameters.

High-level output:
- discrete gait choice: pronking, trotting, bounding, pacing
- continuous residuals: frequency, duration, foot swing height, stance width, body pitch

The current goal is useful condition-aware adaptation under a unified physical objective.
Visible gait-family switching is an important behavior to measure, but it is not the only
success criterion. A unified reward may rationally lead the policy to use a globally robust
gait family and adapt mostly through continuous parameters. If that improves tracking,
stability, energy, impact, and OOD behavior, it is a valid project result rather than an
automatic failure.

Do not force gait differentiation by hard-tuning per-terrain reward weights or gait priors.
Per-terrain reward profiles and score-derived gait priors are diagnostics/ablations, not
the default generalization claim.

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

Current next step as of 2026-06-15:

```text
Use the completed fair gait grid as an offline dataset and re-score it with
several unified, terrain-agnostic reward candidates.
```

Purpose:

```text
Find whether one fixed physical reward weighting over tracking, stability,
slip/scuff, impact, energy, survival, and smoothness can produce reasonable
gait/continuous-parameter trade-offs across all training conditions.
```

The mainline should not yet implement per-terrain reward profiles or a gait
prior. If unified reward training later converges mostly to one gait plus
continuous-parameter adaptation, evaluate that as a possible valid outcome.
Discrete gait switching should be reported as an observed behavior, not assumed
as a required endpoint.

Updated technical route:

```text
1. offline re-score the completed fair grid with unified reward candidates;
2. choose a unified reward from raw metric/Pareto trade-offs, not old gait labels;
3. implement the chosen unified reward and verify offline/live consistency;
4. train PPO without task one-hot and without gait prior as the mainline;
5. evaluate gait ratios, continuous parameter adaptation, performance metrics,
   and OOD behavior;
6. keep task_onehot, selector-only, per-terrain reward, and soft prior as
   diagnostics/ablations.
```

Offline unified-reward re-score has been implemented and run:

```text
scripts/offline_rescore_unified_reward.py
runs/high_level_oracle_gait/fair_target_gait_audit/20260614_training_range_action_grid/unified_reward_rescore
```

Outputs:

```text
summary.md
candidate_selection.md
candidate_weights.json
unified_reward_candidate_stats.csv
unified_reward_decisions.csv
unified_reward_best_by_task_speed.csv
unified_reward_best_by_task_speed_gait.csv
unified_reward_soft_distribution.csv
unified_reward_top1_top2_metric_gaps.csv
```

Readout:

```text
efficiency: primary candidate
balanced: secondary candidate
robustness: diagnostic only
contact_safety: reject as mainline because it collapses to pacing in 14/17 rows
```

Efficiency unified-reward ranking from the offline fair-grid re-score:

```text
flat 0.5: pace > trot > pronk > bound
flat 1.0: trot > pronk > pace > bound
flat 1.5: trot > pace > pronk > bound
flat 2.0: trot > pronk > pace > bound

push 1.2: trot > pronk > bound > pace
push 1.5: trot > pronk > pace > bound
push 1.8: trot > pronk > pace > bound

ramp 0.5: pace > trot > pronk > bound
ramp 1.0: pace > trot > pronk > bound
ramp 1.5: pronk > pace > trot > bound
ramp 2.0: trot > pronk > pace > bound

rough 0.5: pace ~= trot > pronk > bound
rough 1.0: pace > pronk > trot > bound
rough 1.5: pace > trot > pronk > bound
rough 2.0: trot > pace > pronk > bound

stones 1.7: pace > pronk > bound > trot
stones 2.0: pace > trot > bound > pronk
```

Implementation status:

```text
scripts/train_high_level_oracle_ppo.py now supports:
  --reward-profile task_focus_v4
  --reward-profile unified_efficiency
  --reward-profile unified_balanced

scripts/evaluate_fixed_gait_live_reward.py and
scripts/evaluate_gait_target_fairness.py also accept the same flag, so live
audit and training can use matching metric weights.
```

Important caveat:

```text
unified_efficiency is a live proxy for the offline efficiency score. The live
wrapper does not yet expose separate impact/scuff scores, so those parts are
represented through action-health, boundary, clearance, and smoothness terms.
Run a live audit before PPO training and compare its ranking against the
offline re-score.
```

Next implementation/validation step:

```text
Run fixed-gait live reward audit with --reward-profile unified_efficiency. If
the live ranking is not wildly inconsistent with the offline re-score, start a
short no-task PPO run using unified_efficiency. Keep unified_balanced as the
second candidate if efficiency looks too biased in live rollouts.
```

Historical diagnostics and completed audits:

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

Historical fair-audit procedure:

```text
1. scan the actual sampled training command range;
2. for each task, speed, and gait, find that gait's best continuous parameters
   under the same search budget;
3. compare gait families as fairly as possible at each task/speed point;
4. inspect top-vs-second margins, raw metric gaps, and Pareto trade-offs;
5. use the result as diagnostic evidence for unified reward design, not as an
   automatic task-labeled gait target generator.
```

Do not treat the old fixed-residual audit or hand-written task labels as final
proof of the target gait. Do not use the fair search result to force a new
task-labeled gait target by default; use it first to design and audit a unified
terrain-agnostic physical reward.

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

Interpretation rule:

```text
--training-range uses representative points for the sampled training ranges; it
is a diagnostic approximation, not an integral over the continuous command
distribution.

--extended is diagnostic only. Extra speeds outside the current training
distribution should not be used directly as training targets.
```

If gait ranking changes strongly with speed, do not use a single fixed target
gait per task. The next prior/target should become a distribution conditioned on
both terrain condition and command speed:

```text
target_distribution = f(condition, cmd_vx)
```

Only after that audit should the project decide whether to accept the empirical
best gait, modify the performance reward, or build a score-derived soft gait
prior. See `CURRENT_GAIT_ADAPTATION_PLAN.md` for the current decision rules.

Completed sampled-range fair audit:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260614_training_range_action_grid
```

Primary comparison artifacts:

```text
fair_best_continuous_params_comparison.md
fair_best_continuous_params_comparison.csv
fair_task_speed_gait_decision_analysis.md
fair_task_speed_gait_decisions.csv
fair_task_speed_soft_distribution.csv
fair_top1_top2_metric_gaps.csv
```

These files are the intended interpretation layer: for every task-speed-gait,
first choose that gait family's best continuous parameters under the equal
search budget, then compare gait families at those per-gait optima. Do not
interpret the scan as a direct search for a hand-written target gait.

`fair_task_speed_gait_decision_analysis.md` classifies top-vs-second neutral
score margins as:

```text
< 0.01: tie/noise -> keep soft
0.01-0.03: weak advantage -> soft preference
>= 0.03: clearer advantage -> sharp soft or hard only after raw-metric review
```

`fair_top1_top2_metric_gaps.csv` should be used before reward changes. It shows
which raw metrics make the top gait win or lose relative to the second-best gait
after both have received their own best continuous parameters.

Metric-level readout from the top-vs-second gaps:

```text
Many neutral-score winners are not uniformly better. They often win via
lateral/scuff/progress-style terms while losing on energy, impact, or tracking.

Pronk frequently beats trot by lateral/scuff terms, but often costs more energy
and impact.

Trot is clearly healthier at flat 2.0 and rough 2.0, where it improves tracking
and/or efficiency against the runner-up.

Push is not evidence for pacing. It is mostly a pronk-vs-trot trade-off, and
at push 1.8 the live reward already prefers trot over the neutral-score winner.

Stones 1.7 is effectively ambiguous. Stones 2.0 favors pacing on tracking,
impact, and energy, but it pays lateral/fall trade-offs.
```

Coverage:

```text
17 task-speed points:
flat/ramp/rough at 0.5, 1.0, 1.5, 2.0
push_lateral at 1.2, 1.5, 1.8
stepping_stones at 1.7, 2.0

68 task-speed-gait groups * 81 action-space parameter settings = 5508 configs
```

Important readout:

```text
The old hard target labels are not supported as a universal one-hot target when
each gait is first allowed to use its own best continuous parameters.

flat: low-speed pronk/trot are nearly tied; higher speeds favor trot.
ramp: pronk is neutral-score best at all sampled speeds, but trot is close.
rough: pronk is best at 0.5-1.5, trot is best at 2.0.
push: pronk/trot dominate; pacing is not supported by this audit.
stones: pacing dominates, especially at 2.0; bounding is not supported as
        one-hot best.
```

Important 2026-06-15 interpretation correction:

```text
The fair gait audit should not automatically become a task-labeled gait target
generator. Its primary role is diagnostic: after each gait receives an equal
continuous-parameter search budget, it reveals which gait families are Pareto
competitive under terrain-agnostic metrics and which metrics create the
trade-offs.
```

For generalization, the preferred final training objective should be a unified
terrain-agnostic performance reward, not per-terrain reward profiles. Per-terrain
reward weights and score-derived gait priors are both human priors; they may be
useful for diagnostics or controlled ablations, but they should not be treated
as the default final solution if the goal is proprioception-based adaptation to
unseen terrain.

The next design decision is therefore:

```text
Can one unified reward, with fixed weights on physical performance metrics such
as tracking, stability, slip/scuff, impact, energy, and survival, produce useful
condition-dependent gait/parameter choices?
```

If yes, prefer that. If no, only then consider weaker aids, in this order:

```text
1. improve the universal physical metrics;
2. use continuous/observable condition variables rather than task labels;
3. add a weak score-derived soft prior as an ablation, not as the default claim.
```

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
