# Current Gait Adaptation Plan

Date: 2026-06-13

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

The active project goal is useful condition-aware locomotion adaptation under a
unified physical objective. Visible condition-driven gait-family switching is an
important behavior to measure, but it is not the only success criterion.

A unified reward may rationally lead the policy to use a globally robust gait
family and adapt mostly through continuous gait parameters. If that improves
tracking, stability, energy, impact, and OOD behavior, it is a valid project
result rather than an automatic failure.

Do not force gait differentiation by hard-tuning per-terrain reward weights or
gait priors. Per-terrain reward profiles and score-derived gait priors are
diagnostics/ablations, not the default generalization claim.

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

### Target gait validity

The task-map target gait labels are not yet proven to be the true best gait for
each task.

The original target gait labels were chosen from earlier template/score evidence
and engineering intuition. The fixed-gait live reward audit shows that these
labels do not always agree with the current live reward, especially:

```text
push_lateral_pace_recovery: target pacing, live reward prefers trotting
stepping_stones_easy_bound_highspeed: target bounding, live reward prefers pacing
```

This creates a more fundamental question:

```text
Is the target gait wrong, or is the reward/objective failing to express the
desired behavior?
```

Do not tune reward weights to force a target gait until the target gait itself
has been validated under a fair gait-agnostic evaluation.

The fixed-gait live reward audit is useful but not sufficient for this decision,
because it fixes the continuous residuals at zero and therefore compares the
wrapper default gait templates rather than each gait's best achievable behavior.

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

### fixed-gait live reward audit

Run:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845
```

Mode:

```text
fixed gait, continuous residual = 0, selector hold disabled, live reward v4,
full active training speeds
```

Scene and action setup:

```text
num_envs = 32 per fixed task/speed/gait rollout
recorded steps = 1000
warmup steps = 50
samples per row = 32000
terrain_size = 12 m
mesh_type = trimesh
edge_reset_margin = 1.5 m
teleport_thresh = 1.5 m
selector_hold_steps = 0
style_reward_scale = 0.0
```

Active scene definitions:

| task | condition | fixed speeds | terrain/disturbance setup |
|---|---|---|---|
| `flat_trot_efficiency` | `flat` | 0.5/1.0/1.5/2.0 | flat trimesh patch |
| `ramp_up_trot_robustness` | `ramp_up` | 0.5/1.0/1.5/2.0 | ramp heightfield with slope `0.20 * x` |
| `rough_slope_trot_robustness` | `rough_slope` | 0.5/1.0/1.5/2.0 | sloped terrain with slope `0.4 * difficulty` plus random uniform roughness `[-0.05, 0.05]` |
| `push_lateral_pace_recovery` | `push_lateral` | 1.5 | flat terrain plus lateral push every 2.0 s, max velocity 1.5 m/s, random sign |
| `stepping_stones_easy_bound_highspeed` | `stepping_stones_easy` | 2.0 | stepping stones, stone size 0.80 m, gap 0.10 m, platform 1.2 m, depth -0.06 m |

Fixed gait action setup:

```text
continuous residual = 0 for all five residual dimensions
```

This means the audit uses the wrapper's default gait behavior templates, not the
per-row optimized parameters in `gait_template_library.csv`:

| gait | phase | offset | bound | frequency | duration | footswing | stance width | body pitch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pronking | 0.0 | 0.0 | 0.0 | 3.0 | 0.5 | 0.08 | 0.33 | 0.0 |
| trotting | 0.5 | 0.0 | 0.0 | 3.0 | 0.5 | 0.08 | 0.33 | 0.0 |
| bounding | 0.0 | 0.5 | 0.0 | 3.0 | 0.5 | 0.12 | 0.38 | 0.0 |
| pacing | 0.0 | 0.0 | 0.5 | 2.5 | 0.5 | 0.12 | 0.38 | 0.0 |

Therefore this audit answers:

```text
Under the current training reward and wrapper default gait parameters, which
fixed gait receives the highest live reward?
```

It does not answer:

```text
With each gait's separately optimized continuous parameters from the offline
template library, which gait is best?
```

Scene visualization commands:

Use these to visually inspect the same live-audit scenes. Prefer comparing the
target gait against the main competitor.

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

# flat: target trot vs second-best bound
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval flat_trot_efficiency:1.0 --gaits trotting --num-envs 1 --steps 2000 --warmup-steps 0
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval flat_trot_efficiency:1.0 --gaits bounding --num-envs 1 --steps 2000 --warmup-steps 0

# ramp: target trot vs live competitor pronk
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval ramp_up_trot_robustness:1.0 --gaits trotting --num-envs 1 --steps 2000 --warmup-steps 0
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval ramp_up_trot_robustness:1.0 --gaits pronking --num-envs 1 --steps 2000 --warmup-steps 0

# rough: target trot vs live competitor pronk
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval rough_slope_trot_robustness:1.0 --gaits trotting --num-envs 1 --steps 2000 --warmup-steps 0
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval rough_slope_trot_robustness:1.0 --gaits pronking --num-envs 1 --steps 2000 --warmup-steps 0

# push: target pace vs live best trot
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval push_lateral_pace_recovery:1.5 --gaits pacing --num-envs 1 --steps 2000 --warmup-steps 0
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval push_lateral_pace_recovery:1.5 --gaits trotting --num-envs 1 --steps 2000 --warmup-steps 0

# stones: target bound vs live best pace
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval stepping_stones_easy_bound_highspeed:2.0 --gaits bounding --num-envs 1 --steps 2000 --warmup-steps 0
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py --render --no-spawn \
  --eval stepping_stones_easy_bound_highspeed:2.0 --gaits pacing --num-envs 1 --steps 2000 --warmup-steps 0
```

Result summary:

| task | speeds | target | live best | readout |
|---|---|---|---|---|
| `flat_trot_efficiency` | 0.5/1.0/1.5/2.0 | trotting | trotting at all speeds | target agrees with live reward, but margins are modest |
| `ramp_up_trot_robustness` | 0.5/1.0/1.5/2.0 | trotting | pronking at 0.5-1.5, trotting at 2.0 | target is near-tied with pronking; signal is weak |
| `rough_slope_trot_robustness` | 0.5/1.0/1.5/2.0 | trotting | pronking at 0.5-1.5, trotting at 2.0 | target is near-tied with pronking; signal is weak |
| `push_lateral_pace_recovery` | 1.5 | pacing | trotting | live reward contradicts target; pacing ranks last |
| `stepping_stones_easy_bound_highspeed` | 2.0 | bounding | pacing | live reward contradicts target weakly; bounding ranks second |

Key numbers:

```text
target is live best in 6 / 14 task-speed rows
flat: target rank 1/4 at all speeds
ramp: target rank 2/4 at 0.5-1.5, rank 1/4 at 2.0
rough: target rank 2/4 at 0.5-1.5, rank 1/4 at 2.0
push: target rank 4/4, best-target gap = 0.023
stones: target rank 2/4, best-target gap = 0.009
```

Readout:

```text
The live reward audit confirms that reward-only v4 cannot be expected to learn
the task-map target gait on all tasks. For push and stones, PPO choosing trot or
pace is consistent with the current live reward. For ramp/rough, the reward
signal is too close between pronk and trot to provide a clean selector target.
Flat is the only task family where the live reward consistently supports the
task-map target gait.
```

Dominant reward-difference readout:

```text
The live reward is a normalized weighted average:

weighted_metric_reward =
    sum(task_weight_i * score_i) / sum(task_weight_i)

Large absolute weights such as survival can dominate the scale, but they do not
necessarily dominate the gait ranking if all gaits score almost the same. The
ranking is driven by weighted score differences between gaits.
```

Task-specific findings:

- `flat_trot_efficiency`: the reward design is broadly reasonable for the
  intended target. Trotting is live-best at every tested speed, mainly because
  it has better progress/slip/orientation tradeoffs than bound/pace/pronk. The
  margin is modest, so selector learning can still be weak.
- `ramp_up_trot_robustness`: pronking narrowly beats trotting at most speeds,
  with very small gaps. The main differences are progress/yaw/lateral metrics
  versus trotting's smoother/stabler action terms. This is an ambiguous reward,
  not a clean gait target.
- `rough_slope_trot_robustness`: pronking narrowly beats trotting at most
  speeds, mainly through `roll_rate` and `lateral_drift`; trotting often wins
  progress/smoothness/stability. This is also an ambiguous reward.
- `push_lateral_pace_recovery`: trotting beats pacing mainly through
  `lateral_drift` and progress. This means the current push reward is not
  aligned with a pacing target.
- `stepping_stones_easy_bound_highspeed`: pacing beats bounding mainly through
  orientation and progress; bounding does better on lateral drift, and clearance
  does not differentiate because both pace and bound use the same fixed
  footswing height in this audit. The current stones reward is therefore not a
  clean bound objective.

Per-metric weighted gap decomposition:

Generated files:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845/weighted_gap_decomposition.csv
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260612_221845/weighted_gap_decomposition.md
```

The decomposition uses:

```text
raw_gap = target_score - competitor_score
weighted_gap = metric_weight * raw_gap / sum(metric_weights)
```

Positive `weighted_gap` helps the target gait; negative `weighted_gap` helps the
competitor.

Key comparisons:

- flat, trot vs bound: trot wins mostly through orientation, progress, slip,
  and vertical bounce. Bound only slightly helps yaw/lateral terms.
- ramp, trot vs pronk: trot loses narrowly at 0.5-1.5 m/s. Pronk is helped by
  yaw/progress/lateral terms; trot is helped by action smoothness, gait
  stability, and sometimes orientation.
- rough, trot vs pronk: trot loses at 0.5-1.5 m/s mostly because pronk is helped
  by `roll_rate` and `lateral_drift`. Trot is helped by progress/orientation at
  high speed and smoothness/stability.
- push, pace vs trot: pace loses mainly because `lateral_drift` contributes
  `-0.0186` and progress contributes `-0.0051`; yaw terms help pace only weakly.
- stones, bound vs pace: bound loses because orientation contributes `-0.0109`
  and progress contributes `-0.0094`; lateral drift helps bound `+0.0094`.
  Clearance contributes `0.0` because bound and pace both have score 1.0 under
  the default fixed parameters.

## Current Main Doubt

Partly resolved by the fixed-gait live reward audit:

```text
The current live reward v4 only partly agrees with the task-map/template target
gaits.
```

The offline template tables may say one gait is best, but the live PPO training
reward can rank a different gait higher because it uses a different metric set
and normalization. This is now confirmed for push and stones, and partly
confirmed as a weak/ambiguous signal for ramp and rough.

New unresolved question:

```text
The task-map target gaits themselves have not been fairly validated, because the
current fixed-gait audit used default continuous parameters rather than equal
per-gait continuous-parameter search.
```

## Immediate Next Step

Do not continue reward-only selector training, curriculum training, or RMA/no-task
training as the next diagnostic. Also do not implement per-terrain reward v5 or
a gait prior yet.

The immediate next step is offline unified-reward re-scoring:

```text
Use the completed fair gait grid as an offline dataset and re-score all
task-speed-gait-parameter rows with several unified, terrain-agnostic physical
reward candidates.
```

Candidate families:

```text
A. balanced:
   tracking + stability + energy + impact + slip/scuff with comparable weights

B. efficiency-oriented:
   heavier tracking, energy, impact, and action smoothness

C. robustness-oriented:
   heavier survival, orientation, lateral recovery, slip/scuff

D. contact-safety-oriented:
   heavier scuff, clearance/contact safety, impact, slip
```

Selection criteria:

```text
- not whether the result matches old target_gait labels;
- whether raw metrics and Pareto trade-offs are physically reasonable;
- whether the reward collapses every condition to one gait for bad reasons;
- whether continuous parameters vary sensibly with speed/condition;
- whether top1-top2 margins are large enough to provide a learnable signal.
```

If a unified reward prefers mostly one gait plus continuous-parameter adaptation,
do not call that a failure by default. It becomes a valid hypothesis:

```text
Under a unified physical objective, adaptation may emerge primarily through
continuous gait-parameter tuning, with discrete gait switching appearing only
where it is actually useful.
```

Offline unified-reward re-scoring has been implemented and run:

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

Candidate readout:

```text
efficiency: primary candidate. Best overall energy among the candidates,
            relatively low impact, useful margins, and no total gait collapse.

balanced: secondary candidate. Middle-ground trade-off, but some low-speed
          flat/ramp pacing choices need live-audit scrutiny.

robustness: diagnostic only. Best mean vx/fall numbers, but margins are mostly
            tie/noise and energy/impact are worst.

contact_safety: reject as mainline. It overweights contact/scuff/impact enough
                to collapse to pacing in 14/17 task-speed rows.
```

Efficiency ranking by task/speed after per-gait best-parameter selection:

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
audits and PPO training can use matching metric weights.
```

Important caveat:

```text
unified_efficiency is a live proxy for the offline efficiency score. The offline
score used separate impact/scuff scores from the fair-grid raw metrics; the live
wrapper currently represents those pressures indirectly through action-health,
boundary, clearance, and smoothness terms. Therefore a live reward audit is
required before long PPO training.
```

Next validation step:

```text
Run fixed-gait live reward audit with --reward-profile unified_efficiency and
compare the live ranking against the offline efficiency re-score. If the result
is not wildly inconsistent, start a short no-task PPO run with the same reward
profile. Keep unified_balanced as the second candidate if efficiency looks too
biased in live rollouts.
```

Recommended live-audit command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_fixed_gait_live_reward.py \
  --full \
  --gaits all \
  --num-envs 32 \
  --steps 1000 \
  --warmup-steps 50 \
  --reward-profile unified_efficiency \
  --output-dir runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260615_unified_efficiency
```

If this live audit is broadly consistent, short PPO smoke-run command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_high_level_oracle_ppo.py \
  --run-name 20260615_unified_efficiency_notask_smoke \
  --iterations 50 \
  --save-interval 25 \
  --adaptation-coef 0.1 \
  --reward-profile unified_efficiency \
  --style-reward-scale 0.0 \
  --no-oracle-condition-obs
```

The completed fair gait audit remains the diagnostic basis:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260614_training_range_action_grid
```

Historical fair-audit objective:

```text
For each task/speed/gait, evaluate the best achievable task score under an equal
continuous-parameter search budget, using a gait-agnostic task objective.
```

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

Reward adjustment should be downstream of the fair audit, but the next reward
candidate should still be unified and terrain-agnostic unless a later ablation
explicitly tests stronger task priors.

The audit should compare gait families without using `target_gait` as a label
and without hard-coding contact-style preferences. It should answer:

```text
flat/ramp/rough: is trot actually best or only assumed best?
push_lateral: does pace actually recover better than trot/bound/pronk?
stepping_stones: does bound actually cross better than pace/trot/pronk?
```

Fairness requirements:

- each gait gets the same continuous-parameter grid or optimization budget,
- compare the best score per gait, not only wrapper default parameters,
- use the same scenes, seeds, speeds, episode lengths, and disturbance settings,
- report robustness across seeds/push signs/terrain randomization,
- avoid gait-label rewards or phase/contact-style terms in the validation score,
- report raw metrics as well as any weighted score,
- report Pareto trade-offs instead of forcing every task into a single hard
  winner when the trade-off is ambiguous.

The continuous-parameter search must be per terrain/task and per gait:

```text
for each task/speed:
  for each gait family:
    search frequency, duration, footswing height, stance width, body pitch
    with the same parameter budget
    report best score, best parameters, raw metrics, and Pareto candidates
```

This is necessary because the previous fixed-gait audit used only:

```text
continuous residual = 0
```

That setting compares wrapper defaults, not each gait family's best achievable
behavior.

Implementation:

```text
scripts/evaluate_gait_target_fairness.py
```

This script:

- creates one fixed task/speed scene at a time,
- evaluates every requested gait under an equal continuous-parameter search
  budget,
- defaults to `grid-mode action-space`, so every gait receives the same residual
  action grid in the actual high-level policy action space,
- also supports `grid-mode physical` for explicit physical parameter grids,
- reports raw metrics, current live weighted reward, neutral gait-agnostic
  score, best parameters, soft distribution, and Pareto candidates,
- writes intermediate outputs after every batch so a long server run can be
  resumed with `--skip-existing`.

Recommended 4090 full audit command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_gait_target_fairness.py \
  --full \
  --grid-mode action-space \
  --batch-size 96 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full \
  --skip-existing
```

Recommended `nohup` background command:

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

Monitor:

```bash
tail -f runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full/nohup.log
```

4090 utilization note:

```text
batch-size 96, repeats 2 -> 192 envs, recommended starting point for 24GB
batch-size 128, repeats 2 -> 256 envs, try if memory use remains low
batch-size 32, repeats 2 -> 64 envs, conservative fallback
```

Higher batch size reduces the number of parameter batches. Higher repeats
improves per-configuration statistical reliability but also increases env count.
If the bottleneck is CPU/PhysX scheduling rather than GPU memory, more memory
usage may not translate linearly into faster wall-clock time.

Troubleshooting:

```text
If the child process exits with:

argument --freq-residuals: expected one argument

then the server has an old version of `scripts/evaluate_gait_target_fairness.py`.
The fixed version passes negative residual lists to child processes as
`--freq-residuals=-1.0,0.0,1.0` instead of using a separate argv item that starts
with `-`.

If the child process exits with:

Inplace update to inference tensor outside InferenceMode is not allowed

then the server has an old fair-audit script that used `torch.inference_mode()`
while stepping IsaacGym. The fixed version uses `torch.no_grad()` so env buffers
can be reset safely across batches.

If a child process runs for a while and then exits with status 1 near a partial
final batch, update the script. Older versions recreated IsaacGym with a smaller
env count for the final batch. The fixed version pads the final batch and keeps
the sim/env count constant throughout each child process.

Observed old-log signature:

```text
Finished batch 1/3
Finished batch 2/3
AttributeError: 'dict' object has no attribute 'command_curriculum'
```

This was caused by rebuilding the IsaacGym/WTW env inside the same child process
for the partial final batch after global `Cfg` state had already been mutated.
The fixed script avoids the rebuild.
```

Default action-space grid:

```text
frequency residual: -1, 0, 1
duration residual: 0
footswing residual: -1, 0, 1
stance-width residual: -1, 0, 1
body-pitch residual: -1, 0, 1
```

This is 81 parameter settings per gait, 324 settings per task/speed, and 4536
settings for the full 14-row active task-speed audit before repeats.

Main output files:

```text
fair_gait_grid_results.csv       all evaluated parameter settings
best_by_task_speed_gait.csv      best setting per task/speed/gait
best_by_task_speed.csv           best overall setting per task/speed
pareto_front.csv                 non-dominated candidates per task/speed
summary.md                       readable ranking and soft distribution
run_config.json                  exact command/config metadata
```

Completed run:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260613_action_grid_full
```

Completeness check:

```text
fair_gait_grid_results.csv has 4536 data rows
14 task-speed rows * 4 gait families * 81 action-space parameter settings = 4536
best_by_task_speed.csv has 14 data rows
best_by_task_speed_gait.csv has 56 data rows
pareto_front.csv has 1159 data rows
```

Do not judge scan coverage from `best_by_task_speed*.csv`; those files are
compressed summaries. Use `fair_gait_grid_results.csv` for the full scan.

Coverage correction:

```text
20260613_action_grid_full is complete for active task-map training rows, but it
does not cover every sampled command speed for push/stones.
```

Reason:

```text
The task map marks push_lateral training at vx=1.5 only, and stones training at
vx=2.0 only. `--full` used those active map rows.

The trainer expands one-speed tasks at runtime:
push_lateral -> sampled vx range [1.2, 1.8]
stepping_stones_easy -> sampled vx range [1.7, 2.0]
```

Follow-up commands:

```bash
# Actual sampled training-range audit for push/stones edges plus centers.
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/evaluate_gait_target_fairness.py \
  --training-range \
  --grid-mode action-space \
  --batch-size 384 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260614_training_range_action_grid \
  --skip-existing

# Extended diagnostic audit, including push 0.5/1.0/1.5/2.0 and
# stones 1.0/1.5/2.0. Some of these speeds are outside current training.
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/evaluate_gait_target_fairness.py \
  --extended \
  --grid-mode action-space \
  --batch-size 384 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260614_extended_action_grid \
  --skip-existing
```

Interpretation guardrails:

```text
--training-range is a representative-point audit for the sampled training
ranges. It is closer to the actual trainer than --full for push/stones, but it
is still not an integral over the continuous command distribution.

--extended is diagnostic only. Speeds outside the current training distribution
should reveal whether gait preference is speed-dependent, but they should not be
used directly as training targets unless the training curriculum is deliberately
expanded.
```

If the fair audit shows that gait ranking changes with speed, do not use a
single task-level target gait. The target/prior should become:

```text
target_distribution = f(condition, cmd_vx)
```

This is especially important for push and stepping stones, where a fixed
task-level label such as `push -> pace` or `stones -> bound` may be too coarse.

The fair audit may produce a soft distribution rather than a one-hot target:

```text
flat: likely trot-dominant if the margin is robust
ramp/rough: possibly trot/pronk mixture if metrics are close
push: possibly trot/pace mixture depending on recovery metrics
stones: possibly pace/bound mixture depending on crossing, scuffing, and stability
```

The next design decision after that audit is:

```text
A. accept the live reward optimum, even if push becomes trot and stones becomes pace;
B. modify the performance reward so the intended gait is actually live-best;
C. add a score-derived soft gait prior, using template/live score evidence rather
   than hard-coded task labels.
```

### training-range fair audit result

Completed run:

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

These files are the intended interpretation layer. For every task-speed-gait,
the audit first selects that gait family's best continuous-parameter setting
under the equal action-space grid. Gait-family comparison is then made only
between those per-gait optima. This scan should not be interpreted as directly
searching for a pre-existing hand-written target gait.

`fair_task_speed_gait_decision_analysis.md` is the compact decision table. It
uses the following margin heuristic:

```text
neutral margin < 0.01: tie/noise -> keep soft
0.01 <= margin < 0.03: weak advantage -> soft preference
margin >= 0.03: clearer advantage -> sharp soft or hard only after raw-metric review
```

`fair_top1_top2_metric_gaps.csv` is the next reward-design diagnostic. It shows
which raw metrics make the top gait win or lose relative to the second-best gait
after both have received their own best continuous parameters.

Metric-level readout:

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

Neutral-score best gait by task-speed:

```text
These winners are comparisons between each gait's own best continuous-parameter
setting, not wrapper-default gait comparisons.

flat 0.5: pronking, but trotting is very close
flat 1.0: pronking, but trotting is very close
flat 1.5: trotting, pronking close
flat 2.0: trotting, clearer margin

ramp 0.5/1.0/1.5/2.0: pronking, with trotting consistently second

rough 0.5/1.0/1.5: pronking, with trotting close
rough 2.0: trotting

push 1.2/1.5/1.8: pronking, with trotting close; pacing is not supported

stones 1.7/2.0: pacing; bounding is not supported as one-hot best
```

Interpretation:

```text
The old hard task-map labels are not supported as final one-hot targets.
```

Important 2026-06-15 interpretation correction:

```text
Do not automatically convert the fair audit into task-labeled gait targets.
The fair audit is primarily a diagnostic tool. It tells us which gait families
are Pareto competitive after each gait receives its own best continuous
parameters under the same search budget, and which physical metrics create the
trade-offs.
```

For generalization, the preferred final training objective should be a unified
terrain-agnostic performance reward, not per-terrain reward profiles. Per-terrain
reward weights and score-derived gait priors are both human priors. They may be
useful for diagnostics or ablations, but they should not be treated as the
default final solution if the goal is proprioception-based adaptation to unseen
terrain.

The next design question is:

```text
Can one unified reward, with fixed weights on physical performance metrics such
as tracking, stability, slip/scuff, impact, energy, and survival, produce useful
condition-dependent gait/parameter choices?
```

Preferred order:

```text
1. first improve the universal physical metric definitions and weights;
2. only use condition dependence if it comes from continuous/observable physical
   variables rather than task labels;
3. use score-derived soft gait priors only as an ablation or fallback, not as
   the default generalization claim.
```

## Updated Technical Route

Mainline:

```text
1. Offline re-score the completed fair grid with several unified reward
   candidates.
2. Choose 1-2 unified reward candidates based on raw metric trade-offs, not old
   target gait labels.
3. Implement the chosen unified reward in the live wrapper and run a fixed/fair
   reward audit to confirm offline/live consistency.
4. Train PPO with no task one-hot and no gait prior.
5. Evaluate both discrete gait ratios and continuous parameter adaptation on
   in-distribution and OOD terrain.
```

Mainline success criteria:

```text
- improved tracking, stability, survival, energy, impact, slip/scuff, and route
  behavior;
- sensible continuous parameter changes with speed/condition;
- gait-family switching if it naturally helps, but not required everywhere;
- OOD performance does not depend on explicit task labels.
```

Diagnostic/ablation route:

```text
- task_onehot oracle: checks whether the reward/action space can support
  condition-specific behavior when condition information is explicit;
- selector-only: checks discrete selector credit assignment without continuous
  residual shortcuts;
- per-terrain reward profile: upper-bound/diagnostic for task priors, not a
  generalization claim;
- score-derived soft gait prior: weak-supervision ablation or fallback if unified
  reward cannot provide a learnable selector signal.
```

Interpretation rule:

```text
If the mainline learns mostly one gait family plus useful continuous-parameter
adaptation, report that honestly. It may mean discrete gait switching is not
necessary under the chosen unified physical objective, rather than that the
project failed.
```

## Proposed Reward v5 Candidate

Status:

```text
paused, not implemented
```

This candidate is based on the fixed-gait audit and weighted gap decomposition.
It was checked by offline reweighting of the existing audit scores. Under that
offline reweighting, the target gait becomes rank 1 in all 14 active task-speed
rows. This must still be implemented and re-run through live fixed-gait audit
before training.

This proposal is now paused because it assumes the task-map target gait labels
are correct. That assumption is not yet established for push and stepping stones,
and may be ambiguous for ramp and rough.

Do not implement these weights until the fair target-gait audit decides whether
the intended target gaits are valid, should be changed, or should become soft
distributions rather than hard labels.

Important implementation note:

```text
The current reward_focus mechanism is additive and cannot cleanly reduce base
weights such as progress or lateral_drift for a specific task.
```

Reward v5 should therefore use explicit task reward profiles, or an override
mechanism, rather than only adding focus tokens.

Candidate explicit weights:

| task | progress | yaw | orient | roll | yaw_rate | lateral | vertical | slip | clearance | stability | smooth | magnitude | boundary | survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat | 1.0 | 0.3 | 0.3 | 0.0 | 0.0 | 0.8 | 0.8 | 1.2 | 0.0 | 0.4 | 0.7 | 0.6 | 0.8 | 2.0 |
| ramp | 0.8 | 0.0 | 1.5 | 0.0 | 0.0 | 0.1 | 0.0 | 0.4 | 0.0 | 1.5 | 1.8 | 0.6 | 0.8 | 2.0 |
| rough | 1.3 | 0.3 | 1.6 | 0.0 | 0.0 | 0.3 | 0.0 | 0.5 | 0.0 | 0.8 | 1.0 | 0.6 | 0.8 | 2.0 |
| push | 0.6 | 1.0 | 1.5 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.5 | 0.6 | 0.8 | 2.0 |
| stones | 0.4 | 0.8 | 0.2 | 0.0 | 0.0 | 1.4 | 0.0 | 0.0 | 0.2 | 0.6 | 0.7 | 0.6 | 0.8 | 2.0 |

Rationale by task:

- flat: keep v4 unchanged because trot is already live-best at every speed.
- ramp: reduce yaw/lateral/slip terms that help pronk; increase
  orientation/smoothness/stability. This intentionally treats ramp trot as a
  conservative, smooth robustness gait. It is the most aggressive part of v5 and
  should be visually checked carefully.
- rough: remove the `roll_rate` reward because it is the strongest term helping
  pronk over trot; reduce lateral/slip emphasis; increase orientation/progress
  and smoothness/stability.
- push: remove lateral drift as a selector-driving term because it currently
  makes trot beat pace; reduce progress; emphasize yaw/yaw-rate/orientation,
  which are the terms where pace is competitive.
- stones: reduce progress/orientation because they make pace beat bound; increase
  lateral drift and yaw tracking where bound is better. Clearance is kept weak
  because default bound and pace both saturate clearance at 1.0 in the current
  audit.

Offline reweighting readout:

```text
flat: trot rank 1 at 0.5/1.0/1.5/2.0
ramp: trot rank 1 at 0.5/1.0/1.5/2.0
rough: trot rank 1 at 0.5/1.0/1.5/2.0
push: pace rank 1 at 1.5, but only narrowly over pronk
stones: bound rank 1 at 2.0
```

Risk:

```text
This v5 candidate may overfit the fixed default gait templates and may make the
reward less natural, especially on ramp and push. After implementation, the next
mandatory step is another fixed-gait live reward audit before any PPO training.
```

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

Any plan change, validation result, failed run, corrected interpretation, or
new decision rule must be written back to the project documents immediately.
The current project state must not live only in chat history.

When adding a new experiment, its config/summary must explicitly state:

- whether target gait enters the reward,
- `style_reward_scale`,
- selector prior scale, if any,
- whether continuous residuals are trained,
- whether task one-hot is present,
- whether RMA is enabled,
- whether training speed is fixed or sampled over a range,
- whether metrics come from mixed training rollout or independent single-task eval.
