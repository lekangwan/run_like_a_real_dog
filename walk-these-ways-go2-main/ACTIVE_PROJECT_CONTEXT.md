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
scripts/check_high_level_reward_consistency.py
                                         same-trajectory online/offline reward
                                         consistency check
scripts/rescore_fair_grid_live_profiles.py
                                         re-score one completed fair grid with
                                         multiple live reward weight profiles
scripts/select_heldout_gait_configs.py
                                         select and deduplicate top-k configs
                                         for held-out validation cache
scripts/select_metric_sanity_configs.py
                                         select a small representative config
                                         set before metric sanity simulation
scripts/analyze_metric_sanity_audit.py
                                         analyze raw metric/score direction and
                                         compensation warnings from a small
                                         sanity simulation
scripts/collect_high_level_info_path_data.py
                                         collect proprioceptive history, RMA
                                         latents, gait probabilities, and
                                         reference targets for information-path
                                         diagnosis
scripts/analyze_high_level_info_path.py
                                         offline probes for whether history/RMA
                                         latents contain condition information
                                         and whether gait output uses the latent
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

Current next step as of 2026-06-26:

```text
Do not continue reward tuning or longer direct training yet.

The next step is an information-path diagnostic:
  proprioceptive history -> RMA latent -> gait-selection output.

We need to determine whether:
  1. the proprioceptive history itself contains enough condition/speed evidence;
  2. the RMA student/teacher latents encode that evidence;
  3. the gait-selection output actually uses the latent;
  4. the gait-reference training signal is strong enough without direct task id.
```

Purpose:

```text
The v4 physical reward path has already passed consistency, metric sanity,
representative fair-grid, training-range fair-grid, and held-out top-k checks.
Pure v4 performance training did not naturally produce clear discrete gait
separation. A selector-reference diagnostic was therefore added to test whether
explicit gait-selection guidance from the v4 fair-grid/held-out table can train
task-speed-conditioned gait choice.

The coefficient-0.05 full-action run collapsed to all trotting in independent
evaluation. The coefficient-0.15 high-confidence run fixed that problem under
explicit task id: fixed task-speed evaluation matched the reference top gait in
13/17 rows, with 7/7 high-confidence rows matched and confidence-weighted match
0.932.

Therefore the gait-selection reference signal is now confirmed under explicit
task id. The next diagnostic is whether no-task/RMA can recover the same
condition-dependent structure from proprioceptive history.

The no-task/RMA fixed task-speed evaluation has completed. It improves average
tracking metrics but does not preserve the high-confidence gait-selection
structure. The main failure is ramp 1.0/1.5: the reference says pronking, but
the no-task/RMA policy chooses trotting. This points to a condition-recognition
or selector-signal-transfer issue, not to the selector-reference target itself,
because the explicit-task-id version matched all high-confidence rows.

Therefore the current diagnostic should split the failure into four possible
links instead of calling it simply "RMA failed":

```text
1. History not informative enough:
   proprioceptive history cannot classify task/speed/reference gait.

2. History informative, latent not informative:
   the RMA/adaptation module is losing condition information.

3. Latent informative, gait output insensitive:
   the policy has learned a useful latent but the gait selector ignores it.

4. Gait output sensitive but training signal weak:
   the reference signal or training schedule is not strong enough without task id.
```

Run probes before changing architecture:

```text
1. collect fixed-evaluation samples with:
   task id, command speed, proprioceptive history, RMA teacher latent,
   RMA student latent, gait-selection probabilities, selected gait, and
   reference gait distribution;

2. train simple diagnostic classifiers from history/latent to task family,
   speed bin, and reference top gait;

3. compare direct-task-id and no-task/RMA gait-selection probabilities on
   ramp 1.0 and ramp 1.5;

4. test whether replacing or zeroing the latent changes gait-selection
   probabilities.
```
```

The mainline should not yet implement per-terrain reward profiles or a gait
prior as the main generalization claim. The current selector-reference work is
a diagnostic/ablation: it asks whether the network can learn gait selection
when fair-audit-derived gait-selection guidance is provided explicitly.

Updated technical route:

```text
1. define canonical reward metrics shared by online wrapper and offline checks;
2. add explicit impact and scuffing/contact-safety scores;
3. pass a same-trajectory online/offline consistency check;
4. rerun live fair continuous-parameter search under the corrected objective;
5. run held-out top-k validation as a config-level cache;
6. redesign universal metric definitions and compensation structure;
7. run small metric sanity audits on representative configs;
8. only after a candidate passes ranking/raw-metric review, train PPO without
   task one-hot and without gait prior as the mainline;
9. evaluate gait ratios, continuous parameter adaptation, performance metrics,
   and OOD behavior;
10. keep task_onehot, selector-only, per-terrain reward, and soft prior as
   diagnostics/ablations.
```

Current immediate protocol:

```text
Do not run PPO.
Do not promote canonical_efficiency_v4_physical to training yet.
Select the top-k config union from the v4 training-range fair grid, then run
held-out validation on those configs with fresh seeds. Use the held-out cache to
check whether v4's winners and margins survive outside the search seed.
```

New metric-sanity config selection:

```text
runs/high_level_oracle_gait/metric_sanity/20260617_config_selection

source:
  runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/fair_gait_grid_results.csv

task-speed points:
  flat_trot_efficiency:1.0
  ramp_up_trot_robustness:1.0
  rough_slope_trot_robustness:1.0
  push_lateral_pace_recovery:1.5
  stepping_stones_easy_bound_highspeed:2.0

selection:
  score_keys = weighted_metric_reward_mean, neutral_score
  top_k_per_task_speed_gait_per_score = 1
  selected_unique_configs = 37
```

Run the small simulation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/evaluate_gait_target_fairness.py \
  --config-csv runs/high_level_oracle_gait/metric_sanity/20260617_config_selection/metric_sanity_config_requests.csv \
  --eval-from-config \
  --reward-profile canonical_efficiency_candidate \
  --selection-score-key weighted_metric_reward_mean \
  --batch-size 37 \
  --repeats-per-config 4 \
  --steps 500 \
  --warmup-steps 100 \
  --seed 202 \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202
```

Analyze the small simulation:

```bash
python3 scripts/analyze_metric_sanity_audit.py \
  --input runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/fair_gait_grid_results.csv \
  --score-key weighted_metric_reward_mean \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/analysis
```

2026-06-17 small sanity result:

```text
runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202

configs:
  37 requested configs
  5 representative task-speed points
  repeats_per_config = 4
  seed = 202

analysis:
  runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/analysis
```

Score-best gait by live weighted reward:

```text
flat 1.0:   pronking
ramp 1.0:   pronking
rough 1.0:  pronking
push 1.5:   trotting
stones 2.0: pacing
```

Interpretation:

```text
This is not a final gait ranking because it uses only a small representative
config set. It is a metric sanity check.

The result reinforces that canonical_efficiency_candidate is not ready for PPO:
on flat/ramp/rough, live reward still prefers pronking even though trotting is
competitive or better on tracking and average torque in several comparisons.

The analysis reports 9 raw/score direction disagreements among energy, slip,
impact, and scuffing checks. A key reason is that live score columns average
nonlinear per-step scores, while raw columns report averaged penalties. This is
not automatically a code bug, but it means the score cannot be casually
interpreted as "lower average energy/slip". The metric design and aggregation
must be repaired before full fair-grid reruns or PPO.
```

2026-06-17 canonical reward alignment and fair-grid result:

```text
same-trajectory consistency:
  reward_consistency/20260616_canonical_efficiency_candidate_broad
  96 cases, 39 metrics, max_abs_error = 0, passed = true

fair continuous-parameter search:
  runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid
  training-range = true
  grid_mode = action-space
  batch_size = 128
  repeats_per_config = 2
  steps = 500
  warmup_steps = 100
  reward_profile = canonical_efficiency_candidate
  selection_score_key = weighted_metric_reward_mean
```

This fair-grid result compares gait families only after each
task-speed-gait receives the same continuous-parameter search budget.
It is not a fixed-default-parameter comparison.

Canonical efficiency candidate readout:

```text
top gait counts across 17 task-speed points:
  pronking = 12
  pacing = 3
  trotting = 2

flat:  pronking at 0.5/1.0/1.5/2.0
ramp:  pronking at 0.5/1.0/1.5/2.0
rough: pronking at 0.5/1.0/1.5, pacing at 2.0
push:  pronking at 1.2, trotting at 1.5/1.8
stones: pacing at 1.7/2.0

mean top1-top2 margin = 0.0161
```

Canonical balanced candidate was computed by offline re-scoring the same fair
grid with saved `score_<metric>` columns:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/live_profile_rescore

top gait counts:
  pronking = 12
  pacing = 4
  bounding = 1

mean top1-top2 margin = 0.0077
```

Interpretation:

```text
Both canonical candidates are still pronking-dominant. Balanced reduces margins
but does not produce a more convincing physical/gait trade-off. Do not train PPO
with these candidates yet. However, do not assume pronking dominance is wrong
only because it is visually undesirable or reduces gait diversity. First test
whether pronking is genuinely better on raw physical metrics after held-out
validation.

Engineering prior:
  canonical_efficiency_candidate is not currently considered a plausible final
  reward. Given WTW's known trot competence and the usual efficiency/stability
  advantages of trot, a broad pronking preference is a warning sign. Held-out
  top-k validation is a diagnostic to locate the source of this warning, not an
  endorsement of the candidate.
```

Immediate next validation:

```text
Build held-out validation as a configuration-level metrics cache:

  task + speed + gait + parameter config + seed -> raw metrics + score metrics

For every task-speed-gait, take the top-k parameter configs from one or more
offline reward candidates (suggested k=3 or k=5), deduplicate the union, and
re-evaluate only configs that are not already present in the cache. Use new
random seeds, higher repeats, and the same canonical score table. Report
validated best mean, top-3 mean, std, fall rate, and worst-tail/CVaR-style
performance.

Changing only reward weights should not trigger a new full grid or a full
held-out rerun. Re-score cached raw metrics offline and only run additional
held-out rollouts for newly selected, previously unvalidated configs.

If pronking remains dominant with stable raw-metric advantages, then a
pronk-heavy policy may be the truthful result of this unified objective.

If pronking wins only through high variance, score normalization artifacts, or
unphysical metric loopholes, then revise metric definitions/scales/weights.
```

Held-out config selection implemented:

```text
scripts/select_heldout_gait_configs.py
runs/high_level_oracle_gait/heldout_config_selection/20260617_topk_union_k3
runs/high_level_oracle_gait/heldout_config_selection/20260617_topk_union_k5
```

The generated request set uses:

```text
input:
  runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/fair_gait_grid_results.csv

score keys:
  canonical_efficiency_candidate_score
  canonical_balanced_candidate_score
  neutral_score
  weighted_metric_reward_mean

top_k_per_task_speed_gait = 3
selected_unique_configs = 405

top_k_per_task_speed_gait = 5
selected_unique_configs = 658
```

Recommended server command for a first held-out validation pass:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_gait_target_fairness.py \
  --config-csv runs/high_level_oracle_gait/heldout_config_selection/20260617_topk_union_k3/new_heldout_config_requests.csv \
  --eval-from-config \
  --batch-size 64 \
  --repeats-per-config 4 \
  --steps 500 \
  --warmup-steps 100 \
  --reward-profile canonical_efficiency_candidate \
  --selection-score-key weighted_metric_reward_mean \
  --seed 101 \
  --output-dir runs/high_level_oracle_gait/heldout_validation/20260617_topk_union_k3_seed101 \
  --skip-existing
```

2026-06-17 held-out validation result:

```text
runs/high_level_oracle_gait/heldout_validation/20260617_topk_union_k3_seed101

config request set:
  top-k union k=3
  405 configs
  17 task-speed points
  4 gaits
  validation_seed = 101
  repeats_per_config = 4
```

Held-out ranking by canonical efficiency live reward:

```text
fair search original:
  pronking = 12
  pacing = 3
  trotting = 2

held-out seed101:
  pronking = 14
  pacing = 2
  trotting = 1

top-3-config mean on held-out:
  pronking = 14
  pacing = 2
  trotting = 1
```

Readout:

```text
The pronking-dominant landscape survives held-out validation and is not merely
a single best-config max-over-grid artifact for canonical_efficiency_candidate.
Do not use this candidate for PPO.
```

Held-out ranking by the older neutral score:

```text
held-out best config:
  trotting = 10
  pronking = 4
  pacing = 3

held-out top-3-config mean:
  pronking = 8
  trotting = 7
  pacing = 2
```

This shows that the pronking dominance is specific to the canonical live reward
candidate and not an unavoidable property of the held-out trajectories.

Important metric caveat:

```text
The current held-out run exposed a diagnostic-column mismatch: live
score_energy comes from wrapper reward terms, while some extra raw columns such
as transport_cost_proxy and previously torque_penalty_mean were computed from
post-step env buffers in the evaluator. Their direction can disagree.

The weighted_metric_reward ranking is still valid because it comes from wrapper
reward terms, but energy/impact raw-metric interpretation should be repeated
after evaluator raw primitive logging is aligned with wrapper terms.
```

Next metric-design route:

```text
Do not tune weights merely to make trot win. Use trot as a strong plausibility
check for ordinary flat/ramp/rough conditions, because WTW is known to be strong
at trot, but allow push/stones/speed-specific cases to be decided by measured
performance.

Redesign metrics around a constrained/quality-gated objective:

1. survival and velocity tracking are primary task requirements;
2. orientation, slip, impact, and scuffing/contact safety are safety constraints;
3. energy and smoothness should optimize only after basic locomotion quality is
   acceptable, not compensate for poor tracking.
```

Specific metric issues to fix/check:

```text
energy:
  record both mean mechanical power and a stable cost-of-transport proxy;
  avoid rewarding low movement simply because it uses less total energy.

impact:
  prefer landing-event statistics such as impact velocity/force peak,
  impulse, and high percentiles rather than only whole-rollout averages.

slip:
  normalize by meaningful contact time/contact force and check per-distance
  slip, so short-contact gaits are not automatically favored.

progress/tracking:
  prevent energy/slip/contact terms from fully compensating for failing the
  commanded speed.
```

Only if a new metric definition needs primitives not present in existing CSVs
should we rerun simulation. First run small representative metric sanity checks;
rerun full fair grids only after the metric definitions are stable.

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
  --reward-profile canonical_efficiency_candidate
  --reward-profile canonical_balanced_candidate

New reward-alignment code:
  go2_gym/envs/wrappers/high_level_reward_metrics.py
  scripts/check_high_level_reward_consistency.py
```

Important caveat:

```text
unified_efficiency and unified_balanced are historical incomplete live proxies.
canonical_efficiency_candidate and canonical_balanced_candidate include explicit
impact/scuff scores, but are still unvalidated. They must pass same-trajectory
online/offline consistency before any fair-grid rerun or PPO training.
```

Live unified-efficiency audit completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260615_unified_efficiency
```

Readout:

```text
fixed-gait live top counts: pronk 10, trot 3, pace 1
fixed-gait mean margin: 0.013

offline fair-grid re-score using the same live proxy and per-gait best
continuous parameters:
  unified_efficiency top counts: pronk 15, trot 1, pace 1
  unified_balanced top counts: pronk 15, trot 1, pace 1
```

Decision:

```text
Do not start PPO training with the current unified_efficiency live proxy.
The live proxy over-rewards pronk through energy/slip and lacks explicit
impact/scuff/contact-safety terms from the fair-grid analysis.
```

Safety guard:

```text
scripts/train_high_level_oracle_ppo.py now refuses to train with
all diagnostic-only reward profiles by default. Only profiles marked
validated_for_training are accepted without a deliberate diagnostic override.
```

Corrected next implementation/validation route:

```text
Do not simply add two reward terms and rerun. First prove that the offline
analysis objective and the live training reward are the same objective.

1. Define the unified reward metric table from primitive quantities.
   Required families: progress, yaw, orientation, lateral, slip, energy,
   impact, scuff/contact-safety, smoothness, survival, clearance, boundary.

2. Implement online/offline same-trajectory consistency tests.
   Run a small set of fixed task/speed/gait/continuous-parameter combinations,
   save primitive rollout quantities, compute scores online in the wrapper and
   offline from the saved trajectory, and require per-metric/total-reward
   agreement before any expensive audit.

3. Only after same-trajectory consistency passes, rerun live fair continuous-
   parameter search under the corrected objective.

4. Then run live audits on held-out seeds/rollouts.

5. Only after those checks pass, allow PPO training.
```

First consistency-check command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/check_high_level_reward_consistency.py \
  --reward-profile canonical_efficiency_candidate \
  --eval flat_trot_efficiency:1.0,stepping_stones_easy_bound_highspeed:2.0 \
  --gaits pronking,trotting,bounding,pacing \
  --num-envs 4 \
  --steps 20 \
  --warmup-steps 2 \
  --output-dir runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate
```

Implementation note:

```text
scripts/check_high_level_reward_consistency.py spawns one child process per
task/speed/gait/residual case by default. This avoids repeated IsaacGym
environment construction in one Python process, which can pollute the global
Cfg object and produce errors such as a dict-valued config section missing
attributes like command_curriculum.
```

First consistency-check result:

```text
runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate

reward_profile: canonical_efficiency_candidate
cases: 16
  2 task/speed points:
    flat_trot_efficiency:1.0
    stepping_stones_easy_bound_highspeed:2.0
  4 gaits:
    pronking, trotting, bounding, pacing
  2 residual sets:
    zero
    high_clearance

max_abs_error: 0
passed: True
```

Readout:

```text
The online HighLevelGaitWrapper reward terms and offline recomputation from the
same recorded reward primitives now match exactly for the checked cases.
This validates the shared formula/aggregation path for the canonical candidate.
It does not yet replace fair continuous-parameter search, because the check only
covers two task-speed points and two residual sets.
```

Next step:

```text
Run a broader consistency check across all active training-range task/speed
points and a few representative residual sets. If that passes, rerun or
revalidate the fair continuous-parameter search under
canonical_efficiency_candidate.
```

Broad consistency-check result:

```text
runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate_broad

reward_profile: canonical_efficiency_candidate
cases: 96
  6 task/speed points
  4 gaits
  4 residual sets
metrics per case: 39
max_abs_error: 0
passed: True
```

Readout:

```text
The canonical reward formula/aggregation path is now consistent across the
representative task/speed/gait/residual coverage that was checked. It is
reasonable to proceed to live fair continuous-parameter search under
canonical_efficiency_candidate.
```

Decision:

```text
Do not replace the canonical live fair search with an old-grid raw-metric
re-score. The old fair-grid raw metrics remain useful for sanity checks and
rough expectation setting, but they are not final evidence for the corrected
objective. The corrected objective changed the live wrapper metrics, added
explicit impact/scuff scores, and uses per-step online score aggregation. To
claim each gait received an equal chance to find its best continuous parameters
under canonical_efficiency_candidate, rerun the live fair search with
--selection-score-key weighted_metric_reward_mean.
```

Reusable reward-profile evaluation:

```text
A completed live fair search stores `score_<metric>` columns for all canonical
HighLevelGaitWrapper reward metrics. If future unified reward candidates are
only different weights over the same canonical scores, the same fair-search
rollout can be re-scored without rerunning IsaacGym.

Use:
  scripts/rescore_fair_grid_live_profiles.py

This does not apply if the reward candidate changes metric definitions, adds
new primitive measurements that were not recorded, changes the scene/dynamics,
or changes the action grid itself.
```

Important fair-audit implementation update:

```text
scripts/evaluate_gait_target_fairness.py now supports:
  --selection-score-key neutral_score
  --selection-score-key weighted_metric_reward_mean

Use weighted_metric_reward_mean when selecting best continuous parameters for a
live reward profile such as canonical_efficiency_candidate. Otherwise the script
will keep ranking by the old neutral_score.
```

Fair-search runtime note:

```text
In scripts/evaluate_gait_target_fairness.py, the actual IsaacGym env count is
batch_size * repeats_per_config. The `--num-envs` flag is metadata only for this
script. A run with batch_size=384 and repeats_per_config=2 attempted to create
768 envs and segfaulted during PhysX/env initialization on
flat_trot_efficiency vx=0.5. Use batch_size=128 first for the canonical fair
search.
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

## 2026-06-18 Next Metric-Repair Plan

The 37-config sanity audit is a rejection test, not a final gait ranking test.
It is sufficient to reject `canonical_efficiency_candidate` for PPO, but it
does not prove that trotting is always the true optimum.

Current decision:

```text
Do not train PPO.
Do not run another full fair grid yet.
Do not tune weights around canonical_efficiency_candidate.
Repair the metric definitions and aggregation first.
```

Important correction:

```text
The raw/score direction disagreements are not automatically code bugs.
The current reward records averaged nonlinear per-step scores, while many raw
columns are averaged penalties. E[exp(-x)] and exp(-E[x]) are different, so
variance and timing can change gait ranking.
```

Next implementation target:

```text
1. Make tracking and survival the base task terms.
2. Gate quality/efficiency terms by tracking quality without making reward
   sparse at the start of training.
3. Redefine energy diagnostics to record torque penalty, mechanical power,
   and a stabilized transport-cost proxy.
4. Redefine slip with contact-force/contact-time normalization.
5. Redefine impact with landing-event or high-percentile statistics, not only
   rollout averages.
6. Keep the same 37-config sanity set as a regression test after each metric
   change.
7. Only after sanity passes, rerun small grid -> full fair grid -> held-out
   top-k -> PPO.
```

The immediate goal is physical interpretability of the metric table, not making
any specific gait win. For flat/ramp/rough, trot should be a strong plausibility
baseline in this WTW system, but push/stones and speed-dependent cases remain
data-driven.

Implementation update:

```text
canonical_efficiency_v2_candidate has been added as a diagnostic-only reward
candidate. It is not validated for PPO.
```

Changes in v2:

```text
new score terms:
  tracking_gate
  contact_slip
  power_efficiency
  transport_efficiency
  gated_orientation
  gated_lateral_drift
  gated_contact_slip
  gated_power_efficiency
  gated_transport_efficiency
  gated_impact
  gated_scuffing
  gated_action_smoothness

new raw terms recorded by the wrapper/evaluator:
  contact_slip_penalty
  mechanical_power_abs
  transport_cost_proxy
```

The v2 candidate keeps `progress` and `survival` as base terms, and only rewards
secondary quality/efficiency terms through a tracking-quality gate with a
nonzero floor. This is intended to reduce the failure mode where a gait with
poor command tracking wins by looking cheap or safe.

Required next commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/check_high_level_reward_consistency.py \
  --reward-profile canonical_efficiency_v2_candidate \
  --output-dir runs/high_level_oracle_gait/reward_consistency/20260618_canonical_efficiency_v2_candidate
```

If that passes, rerun the same 37-config sanity set:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/evaluate_gait_target_fairness.py \
  --config-csv runs/high_level_oracle_gait/metric_sanity/20260617_config_selection/metric_sanity_config_requests.csv \
  --eval-from-config \
  --reward-profile canonical_efficiency_v2_candidate \
  --selection-score-key weighted_metric_reward_mean \
  --batch-size 37 \
  --repeats-per-config 4 \
  --steps 500 \
  --warmup-steps 100 \
  --seed 203 \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260618_v2_small_sanity_seed203
```

Analyze:

```bash
python3 scripts/analyze_metric_sanity_audit.py \
  --input runs/high_level_oracle_gait/metric_sanity/20260618_v2_small_sanity_seed203/fair_gait_grid_results.csv \
  --score-key weighted_metric_reward_mean \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260618_v2_small_sanity_seed203/analysis
```

2026-06-18 v2 consistency result:

```text
runs/high_level_oracle_gait/reward_consistency/20260618_canonical_efficiency_v2_candidate

reward_profile = canonical_efficiency_v2_candidate
cases = 16
metrics = 54
max_abs_error = 0
passed = True
```

Interpretation:

```text
The v2 online wrapper terms and offline recomputation are exactly consistent
for the checked same-trajectory cases. This only validates formula alignment.
It does not validate reward quality or gait ranking.
```

2026-06-18 v2 small metric sanity result:

```text
runs/high_level_oracle_gait/metric_sanity/20260618_v2_small_sanity_seed203

reward_profile = canonical_efficiency_v2_candidate
configs = 37 requested representative configs
score-best gait counts:
  trotting = 3 / 5 task-speed points
  pronking = 1 / 5
  pacing = 1 / 5
  bounding = 0 / 5

score-best ranking:
  flat_trot_efficiency vx=1.0 -> trotting
  push_lateral_pace_recovery vx=1.5 -> trotting
  ramp_up_trot_robustness vx=1.0 -> pronking
  rough_slope_trot_robustness vx=1.0 -> trotting
  stepping_stones_easy_bound_highspeed vx=2.0 -> pacing
```

Interpretation:

```text
v2 is a clear improvement over canonical_efficiency_candidate: it no longer
shows broad pronking dominance on the 37-config sanity set. However, it is not
validated for full fair grid or PPO.

Two caveats matter:
1. flat and rough are near-ties, so their winner should not be overinterpreted;
2. rough is suspicious because action_boundary_margin gives a large non-physical
   advantage to trotting over pronking. Action-boundary margin is useful as an
   action-health diagnostic, but it should not decide gait-family quality in a
   fair gait audit or core physical reward.

The next allowed step is to revise v2 into a new diagnostic candidate that
separates physical locomotion reward from action regularization, then rerun
consistency and the same 37-config sanity audit. Do not start full fair grid or
PPO yet.

For fair gait audits, rank configs using only R_physical:
  tracking/progress, survival, orientation, lateral control, contact-normalized
  slip, impact, scuffing/contact safety, and one primary energy-efficiency term.

Keep action_boundary_margin, action_magnitude, action_smoothness, and possibly
gait_stability as action-health diagnostics or tiny PPO regularizers, but do
not let them decide gait-family physical ranking.

For PPO later, the intended form is:
  R_total = R_physical + lambda_reg * R_regularization
with lambda_reg small enough that regularization cannot dominate physical task
success.
```

Why v2 changed the sanity outcome:

```text
The old canonical_efficiency_candidate used a mostly linear weighted average:
progress, orientation, slip, energy, impact, scuffing, smoothness, boundary,
and survival could compensate for each other directly. On the small sanity set,
that structure selected pronking on flat/ramp/rough.

v2 changes the mechanism:
1. progress receives a larger weight and becomes the main task term;
2. survival remains a base term;
3. secondary quality/efficiency terms are multiplied by tracking_gate, so poor
   command tracking reduces the value of looking safe/cheap/smooth;
4. contact-normalized slip, mechanical power, and transport-cost proxies are
   recorded and used through gated scores instead of relying only on the old
   torque and slip proxies.

Observed small-sanity effect:
  old canonical candidate: pronking 3/5, trotting 1/5, pacing 1/5;
  v2 candidate:            trotting 3/5, pronking 1/5, pacing 1/5.

The clearest improvement is flat vx=1.0: v2 selects trotting because it has
better progress/tracking and better gated power/transport scores. The old
candidate selected pronking.

The remaining failure is rough vx=1.0: the tiny trotting win is largely caused
by action_boundary_margin, which is not a core physical gait-quality metric.

v3 should not simply zero one coefficient. It should:
1. split physical reward and action regularization;
2. use mechanical power as the primary reward energy term first;
3. keep transport_cost_proxy logged as a diagnostic until its low-speed/push
   behavior is stable;
4. report raw primitive, normalized score, tracking_gate, and weighted
   contribution together for every gated term;
5. treat stones/high-error cases as task-quality failures before declaring a
   relative gait winner meaningful.
```

2026-06-19 v3 implementation update:

```text
Implemented diagnostic profile:
  canonical_efficiency_v3_physical

Status:
  diagnostic_only_unvalidated_candidate

Purpose:
  fair-gait audit physical ranking only. It is not a PPO reward yet.
```

v3 physical reward terms:

```text
progress
yaw_tracking
survival
gated_orientation
gated_lateral_drift
gated_contact_slip
gated_impact
gated_scuffing
gated_power_efficiency
```

Excluded from v3 physical ranking:

```text
action_boundary_margin
action_magnitude
action_smoothness
gait_stability
gated_action_smoothness
transport_efficiency
```

Rationale:

```text
action-health terms remain logged but should not decide gait-family physical
quality. transport_cost_proxy remains logged as a diagnostic until low-speed
and push behavior is stable.
```

`scripts/analyze_metric_sanity_audit.py` now accepts:

```text
--reward-profile canonical_efficiency_v3_physical
```

and writes:

```text
weighted_contribution_decomposition.csv
top_vs_second_contribution_gaps.csv
```

These tables include metric weight, score, raw primitive, tracking_gate, and
weighted contribution/gap so gated terms are not misread as raw physical wins.

2026-06-19 v3 consistency result:

```text
runs/high_level_oracle_gait/reward_consistency/20260619_canonical_efficiency_v3_physical

reward_profile = canonical_efficiency_v3_physical
tasks = flat_trot_efficiency, stepping_stones_easy_bound_highspeed
gaits = pronking, trotting, bounding, pacing
residual_sets = zero, high_clearance
cases = 16
metrics/primitives = 54
max_abs_error = 0
passed = True
```

Interpretation:

```text
The v3 physical online wrapper reward and offline recomputation are exactly
aligned on the checked same-trajectory cases. This validates formula alignment
only. It does not validate reward quality or gait ranking.

The next allowed step is the same 37-config metric sanity audit using
canonical_efficiency_v3_physical, followed by contribution decomposition.
```

2026-06-19 v3 physical small sanity result:

```text
runs/high_level_oracle_gait/metric_sanity/20260619_v3_physical_small_sanity_seed204

reward_profile = canonical_efficiency_v3_physical
configs = same 37 representative sanity configs
seed = 204
analysis = analysis/
```

Score-best ranking:

```text
flat_trot_efficiency vx=1.0 -> pronking
push_lateral_pace_recovery vx=1.5 -> trotting
ramp_up_trot_robustness vx=1.0 -> pronking
rough_slope_trot_robustness vx=1.0 -> pronking
stepping_stones_easy_bound_highspeed vx=2.0 -> pacing
```

Interpretation:

```text
v3 successfully removed action-regularizer contamination from the fair ranking:
rough is no longer decided by action_boundary_margin.

However, v3 physical is not accepted. With action regularizers removed, the
remaining physical score still prefers pronking on flat/ramp/rough. The
contribution decomposition shows that flat pronking loses to trotting on
progress/tracking and mechanical power, but wins through yaw, lateral drift,
contact slip, orientation, impact, and scuffing. This means secondary physical
terms can still compensate for worse command tracking and power.

The next design issue is therefore not action regularization. It is the
compensation structure among physical terms: tracking/progress and energy need
to behave more like primary task requirements or constraints, not just two more
linear terms among many secondary scores.
```

Next direction:

```text
Do not run full fair grid or PPO with v3.
Design the next candidate around tracking-first or constraint-style scoring.
Examples:
  - require a minimum tracking/progress quality before secondary terms matter;
  - increase progress dominance only after checking it does not make reward sparse;
  - reduce lateral/contact/impact/scuffing ability to compensate for worse
    tracking on flat/ramp/rough;
  - keep action regularizers outside physical fair ranking.
```

2026-06-20 v4 design decision:

```text
v4 should not be another linear weight tweak. It should be a tracking-first /
constraint-style physical score.
```

Proposed structure:

```text
R_physical_v4 =
  base_task_terms
  + safety_constraint_terms
  + tracking_gate_strict * efficiency_terms
```

Design:

```text
base_task_terms:
  progress/tracking = primary
  survival = always active
  orientation = moderate
  yaw_tracking = weak

safety_constraint_terms:
  contact_slip, impact, scuffing, lateral_drift
  use thresholded/saturated scores so small differences inside an acceptable
  safety band cannot collectively overpower better tracking and power.

efficiency_terms:
  mechanical_power / power_efficiency only.
  Activate mainly after tracking is acceptable.

diagnostic-only:
  transport_cost_proxy
  action_boundary_margin
  action_magnitude
  action_smoothness
  gait_stability
```

Suggested strict tracking gate:

```text
tracking_gate_strict = clamp((progress - 0.70~0.75) / (1 - 0.70~0.75), 0, 1)
```

Acceptance criterion:

```text
Do not require trot to win by label. Require the winner to be physically
explainable. On ordinary flat/ramp/rough scenes, small secondary advantages in
lateral/contact/impact/scuffing should not override clearly better tracking and
mechanical power.
```

Next implementation:

```text
1. Add threshold/saturation helper scores.
2. Add canonical_efficiency_v4_physical as diagnostic-only.
3. Run consistency.
4. Run the same 37-config sanity audit.
5. Review contribution decomposition before any small/full fair grid.
```

2026-06-20 v4 implementation update:

```text
canonical_efficiency_v4_physical has been implemented as a diagnostic-only
profile in the shared online/offline metric table.
```

Implemented v4 terms:

```text
progress: 5.0
yaw_tracking: 0.2
survival: 2.0
orientation: 0.8
safety_lateral_drift: 0.25
safety_contact_slip: 0.4
safety_impact: 0.4
safety_scuffing: 0.25
strict_gated_power_efficiency: 0.8
```

Implemented structure:

```text
tracking_gate_strict = clamp((progress - 0.75) / 0.25, 0, 1)

safety_lateral_drift = threshold(lateral_drift_score, low=0.25, high=0.50)
safety_contact_slip = threshold(contact_slip_score, low=0.60, high=0.85)
safety_impact = threshold(impact_score, low=0.80, high=0.92)
safety_scuffing = threshold(scuffing_score, low=0.70, high=0.90)

strict_gated_power_efficiency =
  tracking_gate_strict * power_efficiency_score
```

Important exclusions:

```text
transport_cost_proxy remains diagnostic-only.
action_smoothness, action_magnitude, action_boundary_margin, and gait_stability
remain excluded from v4 fair-ranking reward.
```

Validation commands:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/check_high_level_reward_consistency.py \
  --reward-profile canonical_efficiency_v4_physical \
  --output-dir runs/high_level_oracle_gait/reward_consistency/20260620_canonical_efficiency_v4_physical

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/evaluate_gait_target_fairness.py \
  --config-csv runs/high_level_oracle_gait/metric_sanity/20260617_config_selection/metric_sanity_config_requests.csv \
  --eval-from-config \
  --reward-profile canonical_efficiency_v4_physical \
  --selection-score-key weighted_metric_reward_mean \
  --batch-size 37 \
  --repeats-per-config 4 \
  --steps 500 \
  --warmup-steps 100 \
  --seed 205 \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205

python3 scripts/analyze_metric_sanity_audit.py \
  --input runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205/fair_gait_grid_results.csv \
  --score-key weighted_metric_reward_mean \
  --reward-profile canonical_efficiency_v4_physical \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205/analysis
```

Do not promote v4 to PPO until both consistency and the 37-config sanity audit
pass contribution-level review.

2026-06-20 v4 consistency result:

```text
runs/high_level_oracle_gait/reward_consistency/20260620_canonical_efficiency_v4_physical

reward_profile = canonical_efficiency_v4_physical
tolerance = 1e-05
max_abs_error = 0
passed = True
metric comparison rows = 960
```

Interpretation:

```text
The online HighLevelGaitWrapper reward terms and offline recomputation from the
same recorded trajectory primitives agree exactly for v4. This validates formula
consistency only; it does not prove v4 is a good reward. The next step is the
37-config small metric sanity audit and contribution decomposition.
```

2026-06-20 v4 37-config sanity result:

```text
runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205

rows in analysis best-by-gait table = 20
task-speed points = 5
score_best_warning_count = 0

winner counts:
  trotting: 2
  pronking: 2
  pacing: 1
```

Winners:

```text
flat_trot_efficiency vx=1.0 -> trotting
push_lateral_pace_recovery vx=1.5 -> trotting
ramp_up_trot_robustness vx=1.0 -> pronking
rough_slope_trot_robustness vx=1.0 -> pronking
stepping_stones_easy_bound_highspeed vx=2.0 -> pacing
```

Interpretation:

```text
v4 is a clear improvement over v3 for the small sanity set. Flat and push now
select trotting for physically interpretable reasons: better tracking/progress,
orientation and/or power dominate small secondary differences.

Ramp pronking is currently acceptable as a candidate result: tracking is nearly
tied with trotting, while pronking has better yaw, contact slip, impact,
scuffing, and survival; trotting only has better power/orientation.

Rough remains the main ambiguity. Pronking wins, but pacing/trotting are close
or better on some tracking/contact terms. The win is partly from impact,
orientation, and gated power. This should be checked again in the next small
fair grid or held-out review.

Stones remains task-quality weak: pacing is best, but vx_err is still high
around 0.60 m/s, so do not turn this relative winner into a strong gait target.
```

Decision:

```text
v4 passes the first consistency and small-sanity gates, but is still not
validated for PPO. The next reasonable step is a small fair-grid audit, not a
full grid and not PPO. Review rough and stones carefully after that audit.
```

2026-06-21 v4 representative action-grid result:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260620_v4_physical_representative_action_grid

eval points:
  flat_trot_efficiency:1.0
  ramp_up_trot_robustness:1.0
  rough_slope_trot_robustness:1.0
  push_lateral_pace_recovery:1.5
  stepping_stones_easy_bound_highspeed:2.0

grid_mode = action-space
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
seed = 206
```

Winners:

```text
flat vx=1.0   -> trotting, score 0.8816
push vx=1.5   -> trotting, score 0.7721
ramp vx=1.0   -> pronking, score 0.8716
rough vx=1.0  -> pronking, score 0.8382
stones vx=2.0 -> pacing, score 0.5956
```

Interpretation:

```text
The representative action-grid mostly supports v4's design direction.
Flat and push select trotting for primary-task/power reasons. Ramp pronking is
physically explainable because it has better tracking, yaw, impact, and fall
rate, though worse power than trotting. Rough is almost tied: pronking beats
trotting by only about 0.0025, while trotting has better neutral score, power,
fall rate, and orientation. Treat rough as uncertain rather than as a strong
pronk preference. Stones remains task-quality weak: the best gait is pacing,
but vx_err is still about 0.60, so this should not become a strong target.

v4 is still diagnostic-only. Before PPO, run a training-range v4 fair audit to
check whether the same behavior holds across the active sampled speeds, not only
these five representative points.
```

2026-06-21 v4 training-range action-grid result:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid

analysis:
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid/analysis

grid_mode = action-space
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
task_speed_points = 17
seed = 207
```

Winner counts:

```text
trotting: 10
pronking: 6
pacing: 1
bounding: 0
```

Best gait by task-speed:

```text
flat 0.5/1.0/1.5/2.0:
  trotting at all speeds

push 1.2/1.5/1.8:
  trotting at all speeds

ramp 0.5/1.0/1.5:
  pronking
ramp 2.0:
  trotting

rough 0.5/1.0/1.5:
  pronking
rough 2.0:
  trotting

stones 1.7:
  trotting
stones 2.0:
  pacing
```

Interpretation:

```text
The training-range fair grid supports v4 more strongly than the earlier
representative audit. It does not show the old pronking collapse: flat and push
are consistently trotting, ramp/rough switch from pronking at low-mid speed to
trotting at high speed, and stones remains mixed/weak.

Flat and push winners are mainly explained by progress/tracking plus strict
gated power. Ramp pronking is often physically interpretable, especially at
1.0-1.5 where it wins tracking/contact/impact terms. Rough remains less certain:
some pronking wins are small and partly driven by safety terms. Stones should
still be treated as task-quality weak because vx_err remains high.

This is still search-seed evidence, not PPO-ready validation. Next step is a
held-out top-k config validation cache for this v4 training-range grid.
```

2026-06-21 v4 held-out config selection:

```text
runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3

input:
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid/fair_gait_grid_results.csv

top_k_per_task_speed_gait = 3
score_keys = canonical_efficiency_v4_physical_score, weighted_metric_reward_mean, neutral_score
selected_unique_configs = 299
new_uncached_configs = 299
```

By gait:

```text
bounding: 78
pacing: 76
pronking: 74
trotting: 71
```

This is the input for the next held-out validation simulation:

```text
runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3/new_heldout_config_requests.csv
```

2026-06-21 v4 held-out validation seed208:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208

analysis:
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208/analysis
```

Best-config winner counts:

```text
trotting: 9
pronking: 6
pacing: 2
bounding: 0
```

Compared with the search seed207 training-range grid:

```text
unchanged task-speed winners: 14 / 17
changed:
  ramp 2.0:   trotting -> pronking
  rough 1.0:  pronking -> trotting
  stones 1.7: trotting -> pacing
```

Top-3-config mean winner counts:

```text
trotting: 8
pronking: 7
pacing: 2
```

Interpretation:

```text
The held-out result mostly supports v4 stability: flat and push remain
consistently trotting, ramp is mostly pronking, and stones remains pacing/mixed
with high tracking error. The changed points are exactly the previously weak or
ambiguous regions. rough 1.0 is especially uncertain: best config says trotting,
but top-3 mean still says pronking.

Do not train PPO from this single held-out seed yet. Run one more held-out seed
using the same 299-config request set. If the second seed preserves the same
broad structure, v4 can move to a short PPO diagnostic run.
```

2026-06-22 v4 held-out validation seed209:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209

analysis:
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209/analysis
```

Best-config winner counts:

```text
trotting: 9
pronking: 7
pacing: 1
bounding: 0
```

Stable across search207, held208, and held209:

```text
13 / 17 task-speed points have the same winner in all three runs.

stable trotting:
  flat 0.5/1.0/1.5/2.0
  push 1.2/1.5/1.8
  rough 2.0

stable pronking:
  ramp 0.5/1.0/1.5
  rough 1.5

stable pacing:
  stones 2.0
```

Unstable or boundary points:

```text
ramp 2.0:
  search says trotting, both held-out seeds say pronking.

rough 0.5:
  search/held208 say pronking, held209 says trotting by only about 0.0012.

rough 1.0:
  search/held209 best config says pronking, held208 best config says trotting.
  top-3 mean is split across seeds, so treat as uncertain.

stones 1.7:
  search says trotting, held208 says pacing, held209 best config says pronking,
  while held209 top-3 mean says pacing. This remains task-quality weak and
  should not drive a hard gait target.
```

Decision:

```text
v4 passes the second held-out screen at the broad-structure level. It is now
reasonable to run a short PPO diagnostic, still with canonical_efficiency_v4_physical
marked diagnostic-only. The PPO run should be treated as a training-dynamics
test, not final validation.
```

2026-06-22 short PPO diagnostic result:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100

profile = canonical_efficiency_v4_physical
oracle_condition_obs = false
style_reward_scale = 0.0
z_dim = 16
adaptation_coef = 0.1
iterations = 100
```

Training-log readout:

```text
weighted_metric_reward:
  early 0.6564 -> late 0.6764

vx_err:
  early 0.4456 -> late 0.4201

score_progress:
  early 0.6076 -> late 0.6386

z_error:
  iter0 0.0198 -> iter99 0.0024

action_clip_rate:
  about 0
```

Final mixed-rollout gait ratios:

```text
overall:
  pronk 0.316, trot 0.341, bound 0.204, pace 0.140

flat:
  pronk 0.302, trot 0.345, bound 0.205, pace 0.148

ramp:
  pronk 0.306, trot 0.348, bound 0.211, pace 0.135

rough:
  pronk 0.328, trot 0.350, bound 0.184, pace 0.138

push:
  pronk 0.317, trot 0.330, bound 0.205, pace 0.148

stones:
  pronk 0.327, trot 0.329, bound 0.214, pace 0.129
```

Interpretation:

```text
The PPO diagnostic is stable and improves the v4 reward, progress, and vx_err a
little. RMA distillation also converges. There is no action clipping failure.

However, the selector does not show clear condition-driven differentiation in
the mixed rollout. Gait ratios become broadly trot/pronk-heavy but are very
similar across all tasks. This means the reward is learnable as a performance
objective, but 100 iterations of no-task RMA PPO has not yet produced the
condition-specific gait structure predicted by the fair-gait audits.

Do not draw a final conclusion before independent per-task checkpoint
evaluation. Mixed rollout metrics are not enough to judge selector behavior.
```

2026-06-22 independent eval for short PPO diagnostic:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100/independent_eval/20260622_full_iter099

checkpoint:
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100/checkpoints/high_level_000099.pt
```

Readout:

```text
The deterministic per-task evaluation confirms the mixed-rollout concern.
The learned no-task RMA policy is mostly a global trot/pronk mixture and does
not show clear condition-driven gait differentiation.

Dominant gait:
  flat 0.5/1.0/1.5 -> trot, flat 2.0 -> pronk by a tiny margin
  ramp all speeds -> trot
  rough all speeds -> trot
  push 1.5 -> trot
  stones 2.0 -> trot

Average gait ratios by task remain very similar:
  flat:   pronk 0.384, trot 0.421, bound 0.132, pace 0.063
  ramp:   pronk 0.376, trot 0.423, bound 0.128, pace 0.073
  rough:  pronk 0.369, trot 0.425, bound 0.132, pace 0.074
  push:   pronk 0.366, trot 0.394, bound 0.158, pace 0.082
  stones: pronk 0.356, trot 0.378, bound 0.163, pace 0.103
```

Task-quality warning:

```text
vx_err is still high at difficult/high-speed points:
  push 1.5: vx_err 0.511
  stones 2.0: vx_err 0.992
  rough 2.0: vx_err 0.820
  ramp 2.0: vx_err 0.597
```

Decision:

```text
Do not continue directly to a long no-task RMA run.
v4 is a more plausible reward and PPO can improve it, but selector credit
assignment is still weak without task labels. The next diagnostic should test
whether the same v4 reward can produce condition-specific gait choices when the
policy is given oracle task one-hot. If oracle one-hot still does not
differentiate, the issue is not only RMA/no-task inference; it is selector
credit assignment or reward/action coupling.
```

2026-06-22 oracle task-onehot v4 PPO diagnostic:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100

analysis:
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/analysis
```

Protocol:

```text
reward_profile = canonical_efficiency_v4_physical
oracle_condition_obs = true
style_reward_scale = 0.0
adaptation_coef = 0.1
selector_only = false
iterations = 100
```

Training readout:

```text
weighted_metric_reward: 0.6621 -> 0.6770
vx_err:                 0.4314 -> 0.4143
score_progress:         0.6193 -> 0.6432
strict_gated_power:     0.1963 -> 0.2173
done_rate:              0.0176 -> 0.0188
gait_switch_rate:       0.2359 -> 0.2360
action_clip_rate:       0.0000 -> 0.0000
z_error:                0.0081 -> 0.0032
```

Final mixed-rollout gait ratios:

```text
overall:
  pronking 0.228
  trotting 0.363
  bounding 0.258
  pacing 0.150

flat:
  pronking 0.208, trotting 0.344, bounding 0.291, pacing 0.156
ramp:
  pronking 0.229, trotting 0.375, bounding 0.232, pacing 0.165
rough:
  pronking 0.244, trotting 0.347, bounding 0.257, pacing 0.152
push:
  pronking 0.231, trotting 0.386, bounding 0.251, pacing 0.132
stones:
  pronking 0.230, trotting 0.365, bounding 0.260, pacing 0.145
```

Interpretation:

```text
The oracle task-onehot PPO diagnostic is numerically stable and improves v4
reward, progress, and vx_err slightly, with no action clipping. However, the
mixed rollout still does not show clear condition-driven gait differentiation.
It mostly becomes a global trot/bound mixture, and push/stones do not move
toward the fair-audit pacing/bounding alternatives.

Do not conclude from the mixed rollout alone. As with the no-task RMA run, the
next required step is deterministic independent per-task checkpoint evaluation
of high_level_000099.pt.
```

2026-06-22 independent eval for oracle task-onehot v4 diagnostic:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/independent_eval/20260622_full_iter099

checkpoint:
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/checkpoints/high_level_000099.pt
```

Readout:

```text
Every evaluated task-speed point is trotting-dominant:
  flat 0.5/1.0/1.5/2.0 -> trot
  ramp 0.5/1.0/1.5/2.0 -> trot
  rough 0.5/1.0/1.5/2.0 -> trot
  push 1.5 -> trot
  stones 2.0 -> trot

Average gait ratios:
  flat:   pronk 0.277, trot 0.450, bound 0.252, pace 0.020
  ramp:   pronk 0.271, trot 0.457, bound 0.245, pace 0.026
  rough:  pronk 0.249, trot 0.462, bound 0.259, pace 0.030
  push:   pronk 0.253, trot 0.442, bound 0.274, pace 0.032
  stones: pronk 0.213, trot 0.450, bound 0.289, pace 0.048

Overall average:
  pronk 0.261, trot 0.455, bound 0.256, pace 0.028
```

Task-quality warning:

```text
High-speed / hard tasks still have weak command tracking:
  flat 2.0:   vx_err 0.532
  ramp 2.0:   vx_err 0.602
  rough 2.0:  vx_err 0.804
  push 1.5:   vx_err 0.500
  stones 2.0: vx_err 1.012
```

Decision:

```text
Oracle task-onehot did not produce the condition-specific gait structure
predicted by the v4 fair/held-out audits. It made the policy more uniformly
trotting-dominant instead.

Do not continue direct long training of this v4 PPO setup. The bottleneck is no
longer just RMA/no-task inference. It is selector credit assignment and/or the
coupling between discrete gait choice and continuous residual optimization under
the current v4 reward.

The next step should be a smaller mechanism diagnostic, not another long PPO:
compare selector-only, continuous-only/fixed-gait, and possibly slower
selector-update or explicit per-gait candidate evaluation. The goal is to find
why PPO can improve v4 reward while ignoring the fair-audit gait ranking.
```

2026-06-22 selector-only + task-onehot v4 PPO diagnostic:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100

analysis:
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/analysis
```

Protocol:

```text
reward_profile = canonical_efficiency_v4_physical
oracle_condition_obs = true
selector_only = true
style_reward_scale = 0.0
adaptation_coef = 0.1
iterations = 100
continuous residuals executed as zero
```

Training readout:

```text
weighted_metric_reward: 0.6626 -> 0.6672
vx_err:                 0.4297 -> 0.4267
score_progress:         0.6194 -> 0.6265
strict_gated_power:     0.1955 -> 0.1994
done_rate:              0.0174 -> 0.0187
gait_switch_rate:       0.2363 -> 0.2369
action_clip_rate:       0.0000 -> 0.0000
z_error:                0.0074 -> 0.0021
```

Final mixed-rollout gait ratios:

```text
overall:
  pronking 0.261
  trotting 0.251
  bounding 0.252
  pacing 0.236

flat:
  pronking 0.273, trotting 0.246, bounding 0.225, pacing 0.256
ramp:
  pronking 0.263, trotting 0.233, bounding 0.267, pacing 0.237
rough:
  pronking 0.263, trotting 0.282, bounding 0.241, pacing 0.213
push:
  pronking 0.249, trotting 0.251, bounding 0.257, pacing 0.242
stones:
  pronking 0.253, trotting 0.245, bounding 0.268, pacing 0.234
```

Interpretation:

```text
Selector-only training with oracle task-onehot does not reveal a strong v4
selector signal. The reward improves only slightly and the final gait
distribution is close to uniform across tasks. This is a stronger negative
signal than the full-action runs: even when continuous residuals cannot absorb
the improvement, v4 does not clearly push the discrete selector toward the
fair-audit gait structure within 100 iterations.

Run independent eval once for completeness. If it also remains near-uniform,
the next work should move away from longer reward-only PPO and toward an
explicit selector-credit mechanism or a score-derived selector-supervision
ablation.
```

2026-06-22 independent eval for selector-only + task-onehot v4 diagnostic:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/independent_eval/20260622_full_iter099

checkpoint:
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/checkpoints/high_level_000099.pt
```

Readout:

```text
Every evaluated task-speed point is pronking-dominant, but only weakly:
  flat 0.5/1.0/1.5/2.0 -> pronk
  ramp 0.5/1.0/1.5/2.0 -> pronk
  rough 0.5/1.0/1.5/2.0 -> pronk
  push 1.5 -> pronk
  stones 2.0 -> pronk

Average gait ratios:
  flat:   pronk 0.294, trot 0.263, bound 0.231, pace 0.212
  ramp:   pronk 0.289, trot 0.259, bound 0.229, pace 0.223
  rough:  pronk 0.286, trot 0.253, bound 0.239, pace 0.223
  push:   pronk 0.300, trot 0.247, bound 0.227, pace 0.226
  stones: pronk 0.276, trot 0.234, bound 0.254, pace 0.236

Overall average:
  pronk 0.289, trot 0.256, bound 0.234, pace 0.221
```

Task-quality warning:

```text
High-speed / hard tasks are still weak:
  flat 2.0:   vx_err 0.581
  ramp 2.0:   vx_err 0.656
  rough 2.0:  vx_err 0.830
  push 1.5:   vx_err 0.529
  stones 2.0: vx_err 1.002
```

Decision:

```text
This closes the reward-only v4 selector diagnostic. With task-onehot and
continuous residuals fixed to zero, the selector still does not learn the
fair-audit condition structure. It only drifts toward a weak global pronk bias
and remains close to uniform.

Do not run longer reward-only PPO variants of canonical_efficiency_v4_physical
as the next step. The evidence now points to insufficient discrete selector
credit under the performance reward. The next branch should introduce an
explicit selector-credit ablation, such as score-derived soft selector targets
from fair/held-out audits, or a per-gait candidate evaluation / bandit-style
selector update, while keeping unified physical reward as the performance
objective.
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

## 2026-06-22: Selector Reference Training Diagnostic

当前结论：

```text
canonical_efficiency_v4_physical 作为统一物理奖励，可以继续作为运动性能目标；
但只靠这个奖励，无法稳定训练出“不同场景/速度选择不同离散步态”的结果。
```

已经排除的情况：

```text
1. 不直接告诉网络任务，只靠本体历史和连续参数：没有清晰分化。
2. 直接告诉网络任务，同时允许选择步态和调连续参数：仍然没有清晰分化。
3. 直接告诉网络任务，只允许选择步态、连续参数固定为 0：仍然没有跟随 fair audit / 复测结构。
```

其中第 3 点最关键，因为它说明问题不只是连续参数抢走了作用，
而是统一物理奖励给“步态选择器”的训练信号不够清楚。

新增实现：

```text
scripts/build_soft_selector_targets.py
  从 v4 训练速度范围的独立复测结果生成：
  场景 + 速度 -> 四种步态参考概率 + 可信度。

scripts/train_high_level_oracle_ppo.py
  新增 --selector-targets / --selector-aux-coef。
  当它们启用时，只对步态选择输出加入一个小的参考训练项；
  不改变环境奖励，也不直接约束连续参数。
```

已生成参考表：

```text
runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv
```

表的来源：

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208/fair_gait_grid_results.csv
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209/fair_gait_grid_results.csv
```

参考表统计：

```text
17 个场景-速度点
top gait counts:
  pronking 6
  trotting 9
  bounding 0
  pacing 2
low confidence rows (<0.25): 10
```

下一步只做诊断，不作为主结论：

```text
先跑：直接告诉网络任务 + 只训练步态选择 + 小参考训练项。

目的：
  验证当“应该偏向哪种步态”的信号被明确给到步态选择输出时，
  网络是否具备按场景/速度学习步态分布的能力。

如果这个版本仍不分化：
  优先检查参考表映射、训练项系数、步态选择输出是否接线正确。

如果这个版本能分化：
  再打开连续参数，测试连续参数是否会重新压过步态选择。
```

2026-06-23 first selector-reference run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_choose_gait_refprob_coef005_iter100

设置：
  直接告诉网络任务
  只允许选择步态，连续参数固定为 0
  参考概率表：
    runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv
  参考训练项系数：0.05
  iterations: 100
```

训练日志读数：

```text
weighted_metric_reward: 0.6463 -> 0.6820
vx_err:                 0.4450 -> 0.4055
selector reference loss: 1.4056 -> 1.3180
selector predicted entropy: 1.3433 -> 1.1820

final mixed gait ratio:
  pronking 0.330
  trotting 0.345
  bounding 0.153
  pacing 0.172
```

初步解释：

```text
参考训练项已接上，并且让步态输出不再接近完全平均；
但系数 0.05 还没有让混合训练日志明显贴合参考表。

混合日志中：
  flat 约为 pronk/trot 近似并列；
  ramp 偏 pronk，符合参考方向；
  rough 仍然很弱；
  push 偏 trot，符合参考方向；
  stones 没有偏 pace，仍偏 trot。

必须跑独立评测，按固定场景和固定速度逐项检查。
当前本地 shell 缺少 IsaacGym，未能在本机完成独立评测。
```

独立评测已完成：

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_choose_gait_refprob_coef005_iter100/independent_eval/20260623_training_range_iter099
```

固定 17 个训练速度点的结果：

```text
参考表第一步态 vs 实际第一步态：
  匹配 11 / 17
  按参考表可信度加权的匹配度：0.879

按可信度分组：
  可信度 >= 0.5:   7 / 7 匹配
  可信度 < 0.25:   4 / 10 匹配

实际第一步态统计：
  pronking 6
  trotting 11
  bounding 0
  pacing 0
```

逐类读数：

```text
flat:
  0.5 没匹配，实际 pronk/trot 接近但 pronk 更高；
  1.0/1.5/2.0 匹配 trot。

ramp:
  0.5/1.0/1.5 匹配 pronk；
  2.0 未匹配，实际 trot。

rough:
  0.5 匹配 pronk；
  1.0 未匹配，但参考表本身几乎平手；
  1.5 未匹配；
  2.0 匹配 trot。

push:
  1.2/1.5/1.8 全部匹配 trot。

stones:
  1.7/2.0 都未匹配，实际仍偏 trot；
  但这两个参考点可信度低，且任务本身 vx_err 很高。
```

结论：

```text
步态参考训练项已经证明“接线有效”：在参考表可信度高的点，
策略可以跟随场景/速度改变步态选择。

它没有解决低可信度点，尤其 stones 的问题。stones 不能通过简单加大
参考系数强行解释为成功，因为该场景速度跟踪质量很差，参考表本身也弱。

下一步不应继续只训练步态选择；应打开连续参数，检查在可调参数参与后，
高可信度点的步态结构是否还能保住，同时运动性能是否改善。
```

2026-06-23 full-action selector-reference run:

```text
runs/high_level_oracle_gait/20260623_v4_physical_taskonehot_fullaction_refprob_coef005_iter100

设置：
  直接告诉网络任务
  允许选择步态，也允许调连续参数
  参考概率表同上
  参考训练项系数：0.05
  iterations: 100
```

训练日志读数：

```text
weighted_metric_reward: 0.6330 -> 0.6747
vx_err:                 0.4773 -> 0.4281
参考训练项损失:          1.3723 -> 1.2867
步态输出熵:              1.3447 -> 1.2246

final mixed gait ratio:
  pronking 0.280
  trotting 0.371
  bounding 0.170
  pacing 0.180
```

与“只允许选择步态”的上一轮相比：

```text
只允许选择步态：
  final weighted_metric_reward = 0.6820
  final vx_err = 0.4055
  final gait ratios = pronk 0.330, trot 0.345, bound 0.153, pace 0.172

允许连续参数：
  final weighted_metric_reward = 0.6747
  final vx_err = 0.4281
  final gait ratios = pronk 0.280, trot 0.371, bound 0.170, pace 0.180
```

初步解释：

```text
打开连续参数后，训练日志中的运动表现没有改善，反而略弱。
步态结构更偏向全局 trot，坡地/粗糙地中原本应偏 pronk 的结构变弱。

连续参数也没有形成明显的场景差异：
  frequency 约 2.87-2.98
  duration 约 0.514-0.521
  footswing_height 约 0.095-0.097
  stance_width 约 0.347-0.352
  body_pitch 近 0

这说明连续参数目前没有学出有意义的分场景调节。
仍需独立评测固定 17 个速度点，不能只根据混合训练日志下最终结论。
```

独立评测已完成：

```text
runs/high_level_oracle_gait/20260623_v4_physical_taskonehot_fullaction_refprob_coef005_iter100/independent_eval/20260623_training_range_iter099
```

固定 17 个训练速度点的结果：

```text
参考表第一步态 vs 实际第一步态：
  匹配 9 / 17
  按参考表可信度加权的匹配度：0.649

实际第一步态统计：
  pronking 0
  trotting 17
  bounding 0
  pacing 0
```

与“只允许选择步态”的独立评测对比：

```text
只允许选择步态：
  匹配 11 / 17
  可信度加权匹配度 0.879
  实际第一步态：pronking 6, trotting 11

允许连续参数：
  匹配 9 / 17
  可信度加权匹配度 0.649
  实际第一步态：trotting 17
```

运动指标均值对比：

```text
reward_mean:
  choose-only 0.7756
  full-action 0.7381

vx_err_mean:
  choose-only 0.4063
  full-action 0.4016

gait_switch_rate:
  choose-only 0.2003
  full-action 0.1479

mean_gait_dwell_steps:
  choose-only 4.98
  full-action 6.92
```

解释：

```text
连续参数打开后，策略确实更稳定、更少切换，速度误差略低；
但步态结构被全局 trot 吸收了，尤其 ramp 的高可信度 pronk 参考没有保住。

这说明 selector_aux_coef=0.05 在 full-action 条件下太弱，
不足以抵抗连续参数和统一物理奖励共同形成的全局 trot 偏好。

下一步不应进入不告诉任务/RMA 版本。
应先在 full-action + task-onehot 条件下，只监督参考表中高可信度行，
并适度加大步态参考训练项，检查是否能保住高可信度结构。
```

## 2026-06-24: High-Confidence Selector Reference Full-Action Run Finished

已完成：

```text
runs/high_level_oracle_gait/20260624_v4_physical_taskonehot_fullaction_refprob_highconf_coef015_iter100
```

实验含义：

```text
直接告诉网络当前任务；
允许选择步态，也允许调连续参数；
只使用参考表中可信度 >= 0.25 的行；
步态参考训练项系数从 0.05 提高到 0.15。
```

训练日志读数：

```text
weighted_metric_reward: 0.6293 -> 0.6794
vx_err:                 0.4873 -> 0.4100
参考训练项损失:          1.3801 -> 1.2000
参考训练项平均权重:      0.3320 -> 0.3033
参考表熵:                1.1651 -> 1.1678
步态输出熵:              1.3477 -> 1.1818

final mixed gait ratio:
  pronking 0.347
  trotting 0.348
  bounding 0.131
  pacing 0.174
```

按任务拆分的最终混合训练步态比例：

```text
flat:   pronking 0.344, trotting 0.334, bounding 0.130, pacing 0.192
ramp:   pronking 0.474, trotting 0.265, bounding 0.124, pacing 0.137
rough:  pronking 0.352, trotting 0.351, bounding 0.120, pacing 0.177
push:   pronking 0.250, trotting 0.425, bounding 0.136, pacing 0.189
stones: pronking 0.316, trotting 0.365, bounding 0.148, pacing 0.172
```

连续参数读数：

```text
frequency_mean:        2.9207
duration_mean:         0.5189
footswing_height_mean: 0.0919
stance_width_mean:     0.3496
body_pitch_mean:       0.0007
action_clip_rate:      0.0
```

初步判断：

```text
相比 coefficient=0.05 的 full-action 版本，这次混合训练日志中不再是明显
全局 trotting。坡地更偏 pronking，push 更偏 trotting，rough 接近 pronking
和 trotting 并列。这说明“只使用高可信度参考行 + 更大的步态参考训练项”
确实增强了步态选择结构。

但是这仍然只是混合训练日志，不能替代固定任务/固定速度的独立评测。
连续参数仍然接近全局平均值，尚未证明学出了明显的分场景连续参数调节。
```

下一步：

```text
必须对该 run 做固定 17 个训练速度点的独立评测。

重点检查：
  1. 高可信度参考点是否保住；
  2. ramp 1.0 / ramp 1.5 是否真正偏 pronking；
  3. flat / push 是否仍偏 trotting；
  4. stones 这类低可信度点不要强行解释；
  5. 连续参数是否在独立评测中出现有意义的任务/速度差异。
```

独立评测已完成：

```text
runs/high_level_oracle_gait/20260624_v4_physical_taskonehot_fullaction_refprob_highconf_coef015_iter100/independent_eval/20260624_training_range_iter099
```

固定 17 个训练速度点：

```text
参考表第一步态 vs 实际第一步态 = 13 / 17 匹配
按参考表可信度加权的匹配度 = 0.932

可信度 >= 0.25 的参考点：
  7 / 7 匹配

可信度 < 0.25 的参考点：
  6 / 10 匹配

实际第一步态：
  pronking = 6 / 17
  trotting = 11 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

高可信度点全部匹配：

```text
flat 1.5:  ref trotting, actual trotting
flat 2.0:  ref trotting, actual trotting
ramp 1.0:  ref pronking, actual pronking
ramp 1.5:  ref pronking, actual pronking
push 1.2:  ref trotting, actual trotting
push 1.5:  ref trotting, actual trotting
push 1.8:  ref trotting, actual trotting
```

主要失败点：

```text
rough 1.0:  reference trotting, actual pronking, confidence 0.019
rough 1.5:  reference pronking, actual trotting, confidence 0.155
stones 1.7: reference pacing, actual trotting, confidence 0.183
stones 2.0: reference pacing, actual trotting, confidence 0.102
```

These are all low-confidence rows, so they should not be used as strong evidence
against the current selector-reference setup. Stones also remains low-quality in
tracking:

```text
stones 1.7 vx_err = 0.710
stones 2.0 vx_err = 0.902
```

对三轮诊断的固定速度评测对比：

```text
只允许选择步态，系数 0.05：
  匹配 11 / 17
  可信度加权匹配度 0.879
  高可信度点 7 / 7
  实际第一步态：pronking 6, trotting 11
  reward_mean 0.7756
  vx_err_mean 0.4063
  gait_switch_rate 0.2003

允许连续参数，系数 0.05：
  匹配 9 / 17
  可信度加权匹配度 0.649
  高可信度点 5 / 7
  实际第一步态：trotting 17
  reward_mean 0.7381
  vx_err_mean 0.4016
  gait_switch_rate 0.1479

允许连续参数，只用高可信度参考行，系数 0.15：
  匹配 13 / 17
  可信度加权匹配度 0.932
  高可信度点 7 / 7
  实际第一步态：pronking 6, trotting 11
  reward_mean 0.7548
  vx_err_mean 0.3732
  gait_switch_rate 0.0842
```

判断：

```text
这次结果证明：在直接告诉任务编号的条件下，步态参考训练项是有效的。
它不仅保住了高可信度参考点，还比完整动作系数 0.05 明显减少了
“全局 trotting”问题。

同时，这次平均速度误差最低，步态切换率也最低，说明增强步态参考训练项
没有明显破坏速度跟踪，反而让行为更稳定。

但 pacing/bounding 仍没有成为任何固定速度点的实际第一步态。当前参考表
本身也没有 bounding 作为第一步态；pacing 只出现在 stones 的低可信度点，
且 stones 的速度跟踪很差，因此不应强行要求 pacing/bounding 出现。
```

下一步：

```text
进入 no-task/RMA 版本：
  不再直接告诉网络任务编号；
  保留完整动作；
  保留只监督高可信度参考行；
  保留步态参考训练项系数 0.15。

目的：
  判断网络能否从本体历史/RMA 表征里推断场景差异，并复现这次在
  “直接告诉任务编号”条件下已经验证有效的步态结构。
```

## 2026-06-25: No-Task RMA High-Confidence Selector Reference Run Finished

已完成：

```text
runs/high_level_oracle_gait/20260625_v4_physical_notask_rma_fullaction_refprob_highconf_coef015_iter100
```

这次实验做的事情：

```text
不直接告诉网络任务编号；
让网络用本体历史压缩出来的隐变量判断场景；
允许选择步态，也允许调连续参数；
只对参考表中可信度 >= 0.25 的场景/速度给步态参考训练项；
步态参考训练项系数保持 0.15。
```

训练日志读数：

```text
weighted_metric_reward: 0.6259 -> 0.6903
vx_err:                 0.4940 -> 0.3996
步态参考训练项损失:      1.3789 -> 1.2309
步态参考训练项平均权重:  0.3420 -> 0.3238
步态输出熵:              1.3525 -> 1.2156
RMA 蒸馏误差:            0.0180 -> 0.0025

final mixed gait ratio:
  pronking 0.326
  trotting 0.384
  bounding 0.134
  pacing 0.156
```

按任务拆分的最终混合训练步态比例：

```text
flat:   pronking 0.326, trotting 0.380, bounding 0.132, pacing 0.162
ramp:   pronking 0.360, trotting 0.361, bounding 0.124, pacing 0.156
rough:  pronking 0.368, trotting 0.370, bounding 0.126, pacing 0.135
push:   pronking 0.290, trotting 0.401, bounding 0.153, pacing 0.156
stones: pronking 0.284, trotting 0.407, bounding 0.137, pacing 0.172
```

与“直接告诉任务编号”的同配置版本对比：

```text
直接告诉任务编号：
  weighted_metric_reward = 0.6794
  vx_err = 0.4100
  final ratios = pronking 0.347, trotting 0.348, bounding 0.131, pacing 0.174
  ramp mixed top = pronking
  rough mixed top = pronking/trotting near tie

不直接告诉任务编号，用本体历史/RMA：
  weighted_metric_reward = 0.6903
  vx_err = 0.3996
  final ratios = pronking 0.326, trotting 0.384, bounding 0.134, pacing 0.156
  all task families are slightly trotting-dominant in the mixed log
```

初步判断：

```text
这次运动指标更好：平均物理奖励更高、速度误差更低，RMA 蒸馏也正常收敛。

但步态结构比“直接告诉任务编号”的版本弱。混合训练日志中，flat/ramp/rough/
push/stones 最终都略偏 trotting。尤其 ramp 没有像直接任务版本那样明显偏
pronking。

因此，不能只看训练日志说它成功。必须做固定 17 个训练速度点的独立评测。
如果固定评测中 ramp 1.0 / ramp 1.5 的高可信度 pronking 不能保住，就说明
本体历史/RMA 还没有充分替代任务编号。
```

下一步：

```text
对该 run 跑固定 17 个训练速度点的独立评测。

重点检查：
  1. 高可信度参考点是否仍然 7 / 7 匹配；
  2. ramp 1.0 / ramp 1.5 是否保住 pronking；
  3. flat 1.5 / flat 2.0 和 push 1.2 / 1.5 / 1.8 是否保住 trotting；
  4. 与直接告诉任务编号版本相比，可信度加权匹配度是否接近 0.932；
  5. 如果匹配下降，问题更可能是场景识别/RMA 表征，而不是步态参考训练项。
```

独立评测已完成：

```text
runs/high_level_oracle_gait/20260625_v4_physical_notask_rma_fullaction_refprob_highconf_coef015_iter100/independent_eval/20260625_training_range_iter099
```

固定 17 个训练速度点：

```text
参考表第一步态 vs 实际第一步态 = 10 / 17 匹配
按参考表可信度加权的匹配度 = 0.637

可信度 >= 0.25 的参考点：
  5 / 7 匹配

可信度 < 0.25 的参考点：
  5 / 10 匹配

实际第一步态：
  pronking = 3 / 17
  trotting = 14 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

高可信度点结果：

```text
flat 1.5:  ref trotting, actual trotting
flat 2.0:  ref trotting, actual trotting
ramp 1.0:  ref pronking, actual trotting   <-- failed
ramp 1.5:  ref pronking, actual trotting   <-- failed
push 1.2:  ref trotting, actual trotting
push 1.5:  ref trotting, actual trotting
push 1.8:  ref trotting, actual trotting
```

与“直接告诉任务编号”的同配置版本对比：

```text
直接告诉任务编号：
  匹配 13 / 17
  可信度加权匹配度 0.932
  高可信度点 7 / 7
  实际第一步态：pronking 6, trotting 11
  reward_mean 0.7548
  vx_err_mean 0.3732
  gait_switch_rate 0.0842

不直接告诉任务编号，用本体历史/RMA：
  匹配 10 / 17
  可信度加权匹配度 0.637
  高可信度点 5 / 7
  实际第一步态：pronking 3, trotting 14
  reward_mean 0.7539
  vx_err_mean 0.3552
  gait_switch_rate 0.0698
```

判断：

```text
这次固定速度评测说明：不直接告诉任务编号后，平均速度误差更低、切换更少，
但步态选择结构明显弱化。

关键失败不是 stones 这种低可信度点，而是 ramp 1.0 / ramp 1.5 这两个
高可信度点。直接任务编号版本能学成 pronking；不直接告诉任务编号后变成
trotting。

因此，当前瓶颈更像是“本体历史/RMA 表征没有把场景差异传给步态选择输出”，
或者 0.15 的步态参考训练项在不直接告诉任务编号时仍然不够强。

这不是步态参考表本身无效，因为直接告诉任务编号的版本已经证明参考表有效。
```

下一步候选：

```text
不要直接进入更复杂的部署式训练。

优先做小范围诊断：
  1. 检查 RMA 隐变量是否能区分 flat/ramp/rough/push/stones；
  2. 比较 ramp 1.0 / ramp 1.5 上任务编号版本和 no-task/RMA 版本的
     步态输出概率；
  3. 如果 RMA 表征分不开场景，优先改历史表征/蒸馏；
  4. 如果 RMA 表征能分开但步态输出仍不听，尝试更强步态参考训练项或
     分阶段训练。
```
