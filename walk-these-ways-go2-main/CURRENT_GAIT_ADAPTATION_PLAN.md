# Current Gait Adaptation Plan

Date: 2026-07-04

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

Important correction, 2026-07-04:

The current reference-supervised selector run must be treated as a diagnostic
or upper-bound result, not as the final clean generalization result.

It does not feed a direct task id to the policy at inference time, but it does
train the gait selector with a fair-audit-derived gait-reference target. That is
still training-time prior information. It proves that the architecture can use
condition information when given a clear gait-selection signal, but it does not
prove that unified physical reward alone naturally produces the observed gait
differentiation.

The clean main claim must be evaluated without direct task labels and without a
gait-reference supervision table.

Clean coupling update, 2026-07-05:

The next clean test is not another gait-reference run. It keeps:

```text
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
selector-only first, continuous residuals fixed at zero
clean_physics privileged target
physical_state_coef > 0
```

New code path:

```text
--selector-physical-state-input
```

When active, the gait selector reads:

```text
command vx + student RMA latent + predicted clean physical state
```

The predicted clean physical state is detached before entering the gait selector.
This prevents the gait-policy loss from reshaping the physical prediction head
into a hidden gait label. The physical prediction head is still trained by the
generic physical-state loss.

Purpose:

```text
Test whether making the proprioceptive/RMA information pathway more explicit is
enough for the unified physical reward to produce condition-dependent gait
choices, without telling the policy which gait to choose.
```

First 100-iteration run with this hook completed:

```text
runs/high_level_oracle_gait/20260705_v4_clean_physics_notask_zcmdphys_selectoronly_physcoef010_iter100
```

Setup:

```text
no direct task id
no gait-reference supervision table
selector_latent_cmd_only
selector_physical_state_input
selector_only
residual_train_dims = none
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
```

Training-log readout:

```text
reward: 0.6507 -> 0.7249
vx_err: 0.4597 -> 0.3459
adaptation_loss: 0.0453 -> 0.0022
physical_state_loss: 0.1339 -> 0.0422
z_error: 0.0190 -> 0.0019
gait_switch_rate: 0.2263 -> 0.1326

final training-time gait ratios:
  pronking 0.1997
  trotting 0.7168
  bounding 0.0000
  pacing 0.0835
```

Interpretation:

```text
The new input path is functional: the student latent still learns the cleaned
physical target, and policy reward/speed tracking improve during training.

The training-time gait ratios remain strongly trotting-biased. This is not yet
a final result because training-time ratios mix sampled speeds and resets. The
next required step is fixed 17-row independent evaluation of checkpoint
high_level_000099.pt.
```

Fixed 17-row independent evaluation completed:

```text
runs/high_level_oracle_gait/20260705_v4_clean_physics_notask_zcmdphys_selectoronly_physcoef010_iter100/independent_eval/20260705_training_range_iter099
```

Result:

```text
avg reward = 0.8260
avg vx_err = 0.3483
done_rate = 0.0208
gait_switch_rate = 0.0000
top gait counts = trotting 17 / 17
```

Compared with the previous clean no-reference run that used only command speed
plus student RMA latent:

```text
avg reward delta = -0.0002
avg vx_err delta = +0.0008
top-gait decisions unchanged: trotting 17 / 17 in both runs
```

Interpretation:

```text
Adding predicted clean physical state to the gait selector did not produce
condition-dependent gait-family switching. The behavior is essentially the same
as the previous clean all-trotting result.

This means the immediate failure mode is not simply that the gait selector lacked
an explicit physical-state vector. Under the current unified physical reward,
default continuous parameters, no direct task id, and no gait-reference
supervision, stable all-trotting remains the preferred solution.
```

Information-path probe completed:

```text
runs/high_level_oracle_gait/info_path_probe/20260705_clean_physics_zcmdphys_iter099/analysis
```

Probe results:

```text
history -> task test_acc = 0.586
history -> speed test_acc = 0.855
z_student -> task test_acc = 0.553
z_student -> speed test_acc = 0.390
z_teacher -> task test_acc = 0.942
z_teacher -> speed test_acc = 0.331

student vs teacher mean gait-prob diff = 0.0113
student vs zero latent mean gait-prob diff = 0.0173
student vs shuffled latent mean gait-prob diff = 0.0076
```

Interpretation:

```text
The physical-state selector input increases latent sensitivity only slightly,
but not enough to matter behaviorally. Every task-speed row remains trotting-top.

The clean no-reference line now has a consistent result:
  - history contains some condition information;
  - student latent contains only weak/moderate condition information;
  - the gait selector remains weakly sensitive to the latent;
  - canonical_efficiency_v4_physical does not provide enough incentive for
    discrete gait-family switching when default continuous parameters are used.
```

Curriculum diagnostic support, 2026-07-06:

New training options:

```text
--include-task-ids
--task-sampling-weights
```

These options are for clean curriculum diagnostics. They only change the
training distribution:

```text
which task ids / conditions are included
how many parallel environments each selected task receives
```

They do not add:

```text
direct task id observation
gait-reference supervision table
per-terrain reward changes
```

Reason:

```text
The current all-trotting result may come from mixed training: flat / rough /
push / stones provide a large stable trotting basin, while ramp/pronking may be
a narrow local advantage that gets averaged out. A curriculum test can check
whether ramp/pronking can be learned when ramp is isolated first.
```

Planned clean curriculum diagnostic:

```text
Stage 1: ramp only
Stage 2: ramp + flat
Stage 3: ramp + flat + rough
Stage 4: add push / stepping stones
```

Keep constant across stages:

```text
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
selector_latent_cmd_only
selector_physical_state_input
selector_only
residual_train_dims = none
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
```

Interpretation of Stage 1:

```text
If ramp-only still becomes all-trotting, then the current unified reward and
default continuous parameters do not strongly support ramp/pronking.

If ramp-only learns pronking, then the earlier full-mix all-trotting result may
be caused by the large trotting basin in the mixed training distribution.
```

Stage 1 ramp-only training completed:

```text
runs/high_level_oracle_gait/20260706_v4_curriculum_stage1_ramp_only_zcmdphys_selectoronly_physcoef010_iter100
```

Setup:

```text
only ramp_up_trot_robustness is included
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
selector_latent_cmd_only
selector_physical_state_input
selector_only
residual_train_dims = none
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
```

Training-log readout:

```text
iter 000:
  reward = 0.7063
  vx_err = 0.3193
  gait ratios = pronking 0.2617, trotting 0.2305, bounding 0.2788, pacing 0.2290

iter 002:
  reward = 0.7525
  vx_err = 0.2726
  gait ratios = pronking 0.3281, trotting 0.2456, bounding 0.2861, pacing 0.1401

iter 049:
  reward = 0.8144
  vx_err = 0.2039
  gait ratios = pronking 0.0815, trotting 0.7939, bounding 0.1108, pacing 0.0137

iter 099:
  reward = 0.8323
  vx_err = 0.1898
  gait ratios = pronking 0.0239, trotting 0.9570, bounding 0.0190, pacing 0.0000
```

Interpretation from training logs:

```text
The ramp-only curriculum did not preserve pronking during training. Early
exploration briefly used pronking/bounding, but optimization converged strongly
to trotting by iter 099.

This is strong preliminary evidence that, under the current v4 unified physical
reward and default continuous parameters, ramp alone still prefers trotting.
However, this is not yet the final Stage 1 conclusion because training-time
ratios mix all sampled ramp speeds and reset states. The next required step is
fixed-speed independent evaluation on ramp vx = 0.5, 1.0, 1.5, and 2.0.
```

Stage 1 fixed-speed ramp evaluation completed:

```text
runs/high_level_oracle_gait/20260706_v4_curriculum_stage1_ramp_only_zcmdphys_selectoronly_physcoef010_iter100/independent_eval/20260706_ramp_only_iter099
```

Result:

```text
ramp vx=0.5:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.9051
  vx_err = 0.0785

ramp vx=1.0:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8819
  vx_err = 0.1565

ramp vx=1.5:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8529
  vx_err = 0.2961

ramp vx=2.0:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8217
  vx_err = 0.5090
```

Interpretation:

```text
Stage 1 rejects the simple hypothesis that ramp/pronking disappeared only
because flat/rough/push/stones averaged it out during mixed training.

When ramp is isolated, with no direct task id, no gait-reference supervision,
canonical_efficiency_v4_physical, fixed zero continuous residuals, and the
clean physical-state selector path, the learned policy still chooses trotting
at every tested ramp speed.

Therefore continuing directly to Stage 2 ramp+flat is not useful yet. The next
question is narrower:

Does the current live v4 reward with default continuous gait parameters actually
prefer trotting over pronking on ramp, or is the training process failing to find
a pronking solution that would score higher?

The next diagnostic should compare fixed gait families on ramp under the same
live v4 reward and default continuous parameters.
```

Fixed-gait ramp live-reward check completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260706_v4_ramp_default_gait_check
```

This check explicitly used:

```text
reward_profile = canonical_efficiency_v4_physical
style_reward_scale = 0.0
continuous residuals = zero/default gait-template parameters
```

Result by ramp speed:

```text
ramp vx=0.5:
  best = trotting
  trotting reward = 0.902159
  pronking reward = 0.902111
  margin trotting - pronking = 0.000048

ramp vx=1.0:
  best = pronking
  pronking reward = 0.856981
  trotting reward = 0.843003
  margin pronking - trotting = 0.013978

ramp vx=1.5:
  best = pronking
  pronking reward = 0.778408
  trotting reward = 0.769809
  margin pronking - trotting = 0.008599

ramp vx=2.0:
  best = trotting
  trotting reward = 0.688106
  pronking reward = 0.675894
  margin trotting - pronking = 0.012212
```

Interpretation:

```text
The live v4 reward does not simply prefer trotting on every ramp speed. With
fixed default gait parameters, pronking is slightly better at ramp vx=1.0 and
vx=1.5, while trotting is slightly better at vx=0.5 and vx=2.0.

However, the pronking advantage is small. This explains why ramp-only PPO can
still converge to all-trotting: trotting is competitive at every tested speed,
wins at the low/high ends, and likely forms a broader/more stable optimization
basin than the narrow mid-speed pronking advantage.

Therefore the clean curriculum result should not be interpreted as "the reward
has no pronking signal." A more precise interpretation is:
  the pronking signal exists on ramp mid-speeds, but it is too weak/narrow to
  overcome the all-trotting basin in clean reward-only training.
```

Important evaluator correction:

```text
Before 2026-07-06, scripts/evaluate_high_level_policy_by_task.py did not pass
the run's stored reward_profile from args.json into read_task_specs. Therefore,
policy independent-evaluation reward_mean values could default to legacy
task_focus_v4 even for runs trained with canonical_efficiency_v4_physical.

This does not change the gait-ratio conclusions from those evaluations because
reward_profile is not an input to the policy during evaluation. It does mean
that reward_mean values from old independent policy evaluations should not be
directly compared with canonical fixed-gait live-reward audits.

The script has been fixed so future policy evaluations default to the run's
stored reward_profile, with an explicit --reward-profile override available.
The ramp-only policy evaluation should be rerun once with the corrected script
to get canonical reward numbers, although the previous all-trotting gait-ratio
conclusion remains valid.
```

Corrected ramp-only policy evaluation completed:

```text
runs/high_level_oracle_gait/20260706_v4_curriculum_stage1_ramp_only_zcmdphys_selectoronly_physcoef010_iter100/independent_eval/20260706_ramp_only_iter099_canonical
```

Result:

```text
ramp vx=0.5:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.9009
  vx_err = 0.0778

ramp vx=1.0:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8437
  vx_err = 0.1561

ramp vx=1.5:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.7676
  vx_err = 0.2927

ramp vx=2.0:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.6820
  vx_err = 0.5114
```

Comparison to fixed-gait canonical live reward:

```text
policy vs fixed trotting reward:
  vx=0.5: -0.0012
  vx=1.0: +0.0007
  vx=1.5: -0.0022
  vx=2.0: -0.0061

average policy reward = 0.7985
average fixed trotting reward = 0.8008
average fixed pronking reward = 0.8033
```

Interpretation:

```text
The corrected policy evaluation confirms that the learned ramp-only policy is
essentially the fixed trotting policy with zero continuous residuals.

It does not exploit the small fixed-gait pronking advantage at ramp vx=1.0 and
vx=1.5. The average fixed-pronking reward is only about 0.0026 higher than fixed
trotting across the four ramp speeds, so the pronking advantage is real but very
small.

This initially suggested:
  under clean reward-only training, the all-trotting basin is stronger than the
  observed pronking advantage on ramp.

This sparse-speed interpretation was later refined by the dense-speed fixed-gait
check below.
```

Dense-speed fixed-gait ramp check completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260706_v4_ramp_dense_speed_default_gait_check
```

Result:

```text
vx=0.50:
  best = pronking
  pronking reward = 0.903562
  trotting reward = 0.902507
  pronking - trotting = +0.001055

vx=0.75:
  best = pronking
  pronking reward = 0.885472
  trotting reward = 0.876815
  pronking - trotting = +0.008657

vx=1.00:
  best = pronking
  pronking reward = 0.857487
  trotting reward = 0.845114
  pronking - trotting = +0.012373

vx=1.25:
  best = pronking
  pronking reward = 0.823274
  trotting reward = 0.806235
  pronking - trotting = +0.017040

vx=1.50:
  best = pronking
  pronking reward = 0.781948
  trotting reward = 0.769577
  pronking - trotting = +0.012371

vx=1.75:
  best = pronking
  pronking reward = 0.730519
  trotting reward = 0.726492
  pronking - trotting = +0.004027

vx=2.00:
  best = trotting
  pronking reward = 0.669489
  trotting reward = 0.686302
  pronking - trotting = -0.016813
```

Updated interpretation:

```text
The denser sweep corrects the earlier sparse-speed interpretation.

With fixed default gait parameters and canonical_efficiency_v4_physical,
pronking is not merely a tiny two-point mid-speed signal. It is the best fixed
gait on ramp from vx=0.5 through vx=1.75, while trotting wins only at vx=2.0.

However, the margins are still modest:
  - pronking is nearly tied with trotting at vx=0.5;
  - pronking has its largest advantage near vx=1.25, about +0.017 reward;
  - pronking advantage shrinks again by vx=1.75;
  - trotting clearly wins at vx=2.0.

This changes the failure diagnosis:
  the live v4 reward does contain a ramp/pronking preference over most of the
  ramp speed range, but clean PPO training still converges to all-trotting.

Therefore the current issue is less "reward has no ramp-pronking signal" and
more "the ramp-pronking signal is too weak/noisy for unconstrained reward-only
PPO to escape or avoid the all-trotting basin."
```

Selector hold-step diagnostic, 2026-07-06:

Run:

```text
runs/high_level_oracle_gait/20260706_v4_curriculum_stage1b_ramp_only_hold10_zcmdphys_selectoronly_physcoef010_iter100
```

Setup:

```text
only ramp_up_trot_robustness is included
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
selector_latent_cmd_only
selector_physical_state_input
selector_only
residual_train_dims = none
selector_hold_steps = 10
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
```

Training-log readout:

```text
iter 000:
  reward = 0.7123
  vx_err = 0.3503
  gait ratios = pronking 0.2798, trotting 0.1934, bounding 0.3013, pacing 0.2256

iter 049:
  reward = 0.8023
  vx_err = 0.2018
  gait ratios = pronking 0.2642, trotting 0.4839, bounding 0.1245, pacing 0.1274

iter 099:
  reward = 0.7950
  vx_err = 0.2088
  gait ratios = pronking 0.3369, trotting 0.4023, bounding 0.0103, pacing 0.2505
```

Comparison to the previous ramp-only hold=3 run:

```text
hold=3 iter 099:
  reward = 0.8323
  vx_err = 0.1898
  gait ratios = pronking 0.0239, trotting 0.9570, bounding 0.0190, pacing 0.0000

hold=10 iter 099:
  reward = 0.7950
  vx_err = 0.2088
  gait ratios = pronking 0.3369, trotting 0.4023, bounding 0.0103, pacing 0.2505
```

Interpretation from training logs:

```text
Increasing selector_hold_steps from 3 to 10 did prevent immediate collapse to
all-trotting in the training-time gait ratios. However, it did not produce a
clean ramp/pronking solution. The final policy remains mixed, includes a large
pacing fraction, and has worse reward and speed error than the all-trotting
hold=3 ramp-only run.

This means longer gait holding may reduce rapid switching/noisy attribution,
but by itself it is not a successful solution. The checkpoint still needs
fixed-speed ramp evaluation before making a final judgment, because training
ratios mix speeds and reset states.
```

Fixed-speed evaluation of the hold=10 checkpoint completed:

```text
runs/high_level_oracle_gait/20260706_v4_curriculum_stage1b_ramp_only_hold10_zcmdphys_selectoronly_physcoef010_iter100/independent_eval/20260706_ramp_dense_iter099_canonical
```

Result:

```text
vx=0.50:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.9012
  vx_err = 0.0790

vx=0.75:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8764
  vx_err = 0.1113

vx=1.00:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8442
  vx_err = 0.1554

vx=1.25:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8045
  vx_err = 0.2177

vx=1.50:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.7677
  vx_err = 0.2917

vx=1.75:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.7311
  vx_err = 0.3757

vx=2.00:
  top gait = trotting
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.6868
  vx_err = 0.4957
```

Interpretation:

```text
The hold=10 policy is deterministically all-trotting at every ramp speed.

The mixed pronking/trotting/pacing ratios seen during training were therefore
mostly stochastic exploration under a high-entropy selector, not a learned
conditioned gait preference. Once evaluated deterministically, the selector's
top action is still trotting everywhere.

This rejects selector_hold_steps as the main fix. Increasing gait hold length
alone does not make clean reward-only PPO exploit the fixed-gait pronking
advantage on ramp.
```

## 2026-07-07: Training-Time Gait Reinforcement Diagnostic

The ramp-only result raises a stricter question:

```text
A fixed-gait audit can show pronking slightly above trotting on ramp, but PPO
does not update from that audit table. PPO updates from the actions sampled in
its own rollout buffer and from the estimated advantage of those actions.

Therefore, fixed-gait mean reward and training-time action advantage can
disagree.
```

The next diagnostic should check whether ramp-only training actually reinforces
pronking when pronking is sampled.

Implementation added to `scripts/train_high_level_oracle_ppo.py`:

```text
For each sampled gait in the PPO rollout buffer, log:

adv_<gait>_count
adv_<gait>_mean
adv_<gait>_positive_rate
reward_<gait>_mean
return_<gait>_mean
value_<gait>_mean
```

Purpose:

```text
If fixed-gait audit says pronking is slightly better, but training logs show
trotting has higher mean advantage or higher positive-advantage rate, then PPO
is rationally moving toward trotting under its own state distribution.

If pronking has higher advantage in the rollout buffer but the policy still
collapses to trotting, then the issue is more likely selector optimization,
exploration, entropy, or action-transition dynamics.

If pronking is rarely sampled after early iterations, the fixed-gait signal may
be too narrow or too small for clean PPO to discover reliably.
```

This diagnostic does not replace the need for a same-initial-state paired
comparison. It is the lower-cost first step because it directly asks what the
training algorithm is reinforcing.

First ramp-only advantage-log run completed:

```text
runs/high_level_oracle_gait/20260707_v4_ramp_only_advlog_zcmdphys_selectoronly_physcoef010_iter050
```

Late-window summary over iterations 40-49:

```text
reward avg = 0.7543
vx_err avg = 0.2601

executed gait ratios:
  pronking 0.272
  trotting 0.426
  bounding 0.083
  pacing   0.219

executed-gait grouped advantage:
  pronking +0.0088, positive rate 0.595
  trotting -0.0103, positive rate 0.595
  bounding -0.0282, positive rate 0.569
  pacing   +0.0200, positive rate 0.608
```

Interpretation:

```text
The first result does not support the simple explanation that PPO moves toward
trotting because the rollout buffer gives trotting a stronger advantage.

However, the logged `adv_<gait>` columns were grouped by the action that the
wrapper actually executed after selector-hold logic. PPO log-prob updates are
attached to the action sampled by the policy before selector-hold logic.

Therefore this first run is informative but not yet decisive: selector-hold can
make sampled gait and executed gait differ.
```

Code correction added:

```text
sampled_executed_gait_mismatch_rate

sampled_adv_<gait>_count
sampled_adv_<gait>_mean
sampled_adv_<gait>_positive_rate
sampled_reward_<gait>_mean
sampled_return_<gait>_mean
sampled_value_<gait>_mean

executed_adv_<gait>_count
executed_adv_<gait>_mean
executed_adv_<gait>_positive_rate
executed_reward_<gait>_mean
executed_return_<gait>_mean
executed_value_<gait>_mean
```

The next rerun should use the same ramp-only setup, then compare sampled-gait
advantage with executed-gait advantage. If these disagree strongly, selector
hold is part of the training-signal mismatch.

Rerun with sampled-vs-executed logging completed:

```text
runs/high_level_oracle_gait/20260707_v4_ramp_only_sampled_exec_advlog_zcmdphys_selectoronly_physcoef010_iter050
```

Window summaries:

```text
early iters 0-9:
  reward 0.7484
  vx_err 0.2654
  sampled/executed gait mismatch 0.504

late iters 40-49:
  reward 0.7617
  vx_err 0.2453
  sampled/executed gait mismatch 0.473

late executed gait ratios:
  pronking 0.301
  trotting 0.243
  bounding 0.338
  pacing   0.118

late sampled-gait grouped advantage:
  pronking -0.0073
  trotting -0.0225
  bounding +0.0308
  pacing   -0.0364

late executed-gait grouped advantage:
  pronking -0.0008
  trotting -0.0410
  bounding +0.0275
  pacing   +0.0076
```

Interpretation:

```text
The sampled/executed mismatch is very large, around 0.47-0.50. This means that
nearly half of the selector actions sampled by the policy are not the gait that
the wrapper executes on that step.

Therefore selector_hold_steps=3 is not just a gait-stability mechanism. In the
current PPO implementation it also creates an action-credit mismatch: PPO stores
the newly sampled action and log-prob, while the reward may come from a gait
chosen on an earlier step and held by the wrapper.

This makes the ramp-only result hard to interpret. The current logs do not show
a clean training-time pronking advantage. In the late window, the strongest
sampled-action advantage is bounding, not pronking or trotting.
```

Next diagnostic:

```text
Run the same ramp-only setup once with selector_hold_steps=0. This removes the
sampled/executed mismatch and tests whether clean action-reward attribution
produces a pronking signal under the same reward.

If hold=0 produces a clearer pronking advantage, the hold mechanism or PPO
buffer accounting needs redesign.

If hold=0 still does not produce pronking advantage, the fixed-gait pronking
edge is probably too weak/noisy for this clean PPO setup.
```

Hold=0 rerun completed:

```text
runs/high_level_oracle_gait/20260708_v4_ramp_only_hold0_advlog_zcmdphys_selectoronly_physcoef010_iter050
```

Window summaries:

```text
early iters 0-9:
  reward 0.7062
  vx_err 0.3083
  sampled/executed gait mismatch 0.000
  gait ratios:
    pronking 0.301
    trotting 0.282
    bounding 0.155
    pacing   0.263
  sampled-action advantage:
    pronking -0.0329
    trotting +0.0109
    bounding +0.0213
    pacing   +0.0134

mid iters 20-29:
  reward 0.7153
  vx_err 0.3060
  sampled/executed gait mismatch 0.000
  gait ratios:
    pronking 0.095
    trotting 0.532
    bounding 0.241
    pacing   0.131
  sampled-action advantage:
    pronking -0.0181
    trotting +0.0087
    bounding +0.0083
    pacing   -0.0374

late iters 40-49:
  reward 0.7560
  vx_err 0.2678
  sampled/executed gait mismatch 0.000
  gait ratios:
    pronking 0.092
    trotting 0.770
    bounding 0.074
    pacing   0.064
  sampled-action advantage:
    pronking -0.0280
    trotting +0.0157
    bounding -0.0491
    pacing   -0.0929
```

Interpretation:

```text
With selector_hold_steps=0, the action-credit mismatch disappears.

Under clean sampled-action attribution, the ramp-only PPO rollout consistently
gives pronking negative advantage and trotting positive advantage. The policy
therefore moves toward trotting for a valid PPO reason.

This resolves the immediate contradiction:
  fixed-gait audit:
    forcing pronking from reset can be slightly better than forcing trotting at
    many ramp speeds;

  PPO training:
    under the policy-induced state and speed distribution, sampled pronking has
    worse estimated advantage than sampled trotting.

The fixed-gait pronking edge is therefore not strong enough, stable enough, or
aligned enough with the training rollout distribution to drive clean PPO toward
pronking.
```

Remaining unknown:

```text
The current advantage statistics are averaged over ramp speeds from 0.5 to 2.0.
Dense fixed-gait audit showed pronking has its clearest edge only in the middle
speed range and loses at vx=2.0.

The next diagnostic should split training-time gait advantage by command-speed
bins. This will tell whether pronking is negative everywhere during training,
or only becomes negative after mixing low/mid/high ramp speeds together.
```

Implemented speed-binned training diagnostic in
`scripts/train_high_level_oracle_ppo.py`:

```text
sampled_vx_low_0p50_1p00_<gait>_count
sampled_vx_low_0p50_1p00_<gait>_adv_mean
sampled_vx_low_0p50_1p00_<gait>_adv_positive_rate
sampled_vx_low_0p50_1p00_<gait>_reward_mean

sampled_vx_mid_1p00_1p50_<gait>_count
sampled_vx_mid_1p00_1p50_<gait>_adv_mean
sampled_vx_mid_1p00_1p50_<gait>_adv_positive_rate
sampled_vx_mid_1p00_1p50_<gait>_reward_mean

sampled_vx_high_1p50_2p00_<gait>_count
sampled_vx_high_1p50_2p00_<gait>_adv_mean
sampled_vx_high_1p50_2p00_<gait>_adv_positive_rate
sampled_vx_high_1p50_2p00_<gait>_reward_mean
```

Purpose:

```text
Dense fixed-gait audit showed the ramp pronking edge is speed-local:
  - weak at low speed,
  - clearest around 1.0-1.5,
  - gone or reversed near 2.0.

If training-time pronking is negative even in the 1.0-1.5 speed bin, the clean
PPO rollout distribution does not support pronking despite the fixed-gait audit.

If pronking is positive only in the 1.0-1.5 bin but negative elsewhere, then
full-range ramp training is averaging out a narrow speed-local gait advantage.
```

Speed-binned hold=0 rerun completed:

```text
runs/high_level_oracle_gait/20260708_v4_ramp_only_hold0_speedbin_advlog_zcmdphys_selectoronly_physcoef010_iter050
```

Overall:

```text
sampled/executed gait mismatch = 0

all iterations:
  reward 0.7424
  vx_err 0.2836
  gait ratios:
    pronking 0.076
    trotting 0.628
    bounding 0.164
    pacing   0.133

late iters 40-49:
  reward 0.8117
  vx_err 0.2150
  gait ratios:
    pronking 0.010
    trotting 0.961
    bounding 0.015
    pacing   0.015
```

Training-time sampled-action advantage by speed bin:

```text
all iterations, low speed 0.5-1.0:
  pronking +0.2074
  trotting +0.1376
  bounding +0.1983
  pacing   +0.2273

all iterations, mid speed 1.0-1.5:
  pronking -0.0758
  trotting -0.0576
  bounding -0.0889
  pacing   -0.1015

all iterations, high speed 1.5-2.0:
  pronking -0.3318
  trotting -0.1436
  bounding -0.3023
  pacing   -0.3006

late iters 40-49, low speed 0.5-1.0:
  pronking -0.0903
  trotting +0.0741
  bounding +0.0232
  pacing   -0.0344

late iters 40-49, mid speed 1.0-1.5:
  pronking -0.1767
  trotting -0.0508
  bounding -0.0482
  pacing   -0.1278

late iters 40-49, high speed 1.5-2.0:
  pronking -0.2050
  trotting -0.0827
  bounding -0.1188
  pacing   -0.3204
```

Interpretation:

```text
Speed mixing is not the main explanation.

The fixed-gait dense audit suggested pronking's clearest ramp advantage should
appear around vx=1.0-1.5. But in the clean hold=0 PPO rollout, pronking is more
negative than trotting in that same mid-speed bin.

By the late training window, trotting receives the best or least-bad advantage
in every speed bin, and the policy becomes almost entirely trotting.

Therefore the contradiction is not simply:
  "mid-speed pronking signal got averaged away by low/high speeds."

It is more likely:
  fixed-gait audit and PPO rollout are evaluating different state
  distributions, transition histories, or reward-normalization effects.
```

Next diagnostic decision:

```text
Stop ordinary ramp-only PPO reruns for now.

The next useful check is a paired same-state ramp comparison:
  for the same sampled initial condition and command speed,
  force trotting and pronking separately,
  then compare reward, raw metrics, normalized scores, and weighted
  contributions.

This should focus first on the mid-speed band around vx=1.0-1.5, because that
is where fixed-gait audit says pronking should be most competitive but PPO
training says pronking has worse advantage.
```

Implemented first paired-state audit tool:

```text
scripts/evaluate_paired_gait_live_reward.py
```

Design:

```text
1. Create one high-level environment batch.
2. Reset and set one command speed.
3. Optionally run a context gait first, e.g. trotting for 20 high-level steps.
4. Save the simulator state, high-level wrapper state, low-level observation
   history, command buffers, and relevant robot buffers.
5. Restore that exact saved state and force gait A.
6. Restore the same saved state again and force gait B.
7. Compare gait A - gait B per environment.
```

Default first use:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
reward_profile = canonical_efficiency_v4_physical
```

Why use a trotting context:

```text
PPO training quickly creates trotting-dominant states. The fixed-gait audit from
reset may overestimate pronking if pronking is only good when executed from the
start. A trotting-context paired audit asks whether switching to pronking from a
trotting-like state is actually beneficial.
```

Outputs:

```text
paired_env_metrics.csv
  per-env metrics for gait A and gait B

paired_metric_deltas.csv
  mean/median/std and positive-rate of each metric delta

summary.md
  compact table for reward, tracking, selected raw metrics, normalized scores,
  and contact/safety terms
```

Smoke-test fix:

```text
The first paired audit smoke test failed because the script used
torch.inference_mode() while stepping the environment. This created inference
tensors inside the simulator/wrapper state, and later state restoration failed
on inplace tensor copies.

The script now uses torch.no_grad() for rollout execution. This still avoids
gradient tracking but keeps simulator tensors writable for snapshot/restore.
```

First paired smoke test completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260708_ramp_mid_pronk_vs_trot_context_trot_smoke
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 16
steps = 80
warmup_steps = 20
delta = pronking - trotting
```

Key result:

```text
weighted_metric_reward:
  pronking 0.821691
  trotting 0.790202
  delta +0.031489
  P(delta > 0) = 0.750

vx_err_mean:
  pronking 0.200303
  trotting 0.247980
  delta -0.047677

score_progress:
  pronking 0.832492
  trotting 0.790395
  delta +0.042097

score_yaw_tracking:
  pronking 0.752774
  trotting 0.659505
  delta +0.093269

score_contact_slip:
  pronking 0.812034
  trotting 0.737198
  delta +0.074836

score_power_efficiency:
  pronking 0.521937
  trotting 0.542656
  delta -0.020719

mechanical_power_abs:
  pronking 200.344620
  trotting 189.456406
  delta +10.888225
```

Interpretation:

```text
The paired smoke test supports pronking over trotting for sustained execution
from a trotting-context ramp state at vx=1.25.

This is the opposite of the hold=0 PPO sampled-action advantage, where sampled
pronking was negative in the 1.0-1.5 speed bin.

Therefore the current contradiction is no longer just fixed-reset audit vs
training. It is now:

  sustained forced pronking from the same state can be better;
  but one-step sampled pronking inside PPO has negative advantage.

This points toward a temporal credit/action-duration problem:
  the useful effect of a gait may require sustained execution over multiple
  high-level steps, while PPO currently assigns advantage to single sampled
  steps. The previous wrapper-level selector hold created sampled/executed
  mismatch, so the cleaner fix would be option-style or decision-interval PPO,
  not simply holding inside the wrapper while still sampling every step.
```

Status:

```text
This is still a smoke test with only 16 envs and short rollout length. It should
be repeated with more envs, longer rollout, and several repeats before changing
training architecture.
```

Paired confirmation run completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260708_ramp_mid_pronk_vs_trot_context_trot_confirm
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 64
repeats = 3
steps = 300
warmup_steps = 50
delta = pronking - trotting
```

Main result:

```text
weighted_metric_reward:
  pronking 0.824598
  trotting 0.806790
  delta +0.017808
  median delta +0.016536
  P(delta > 0) = 0.760

By repeat:
  repeat 0: mean delta +0.01784, P(delta>0)=0.734
  repeat 1: mean delta +0.01706, P(delta>0)=0.734
  repeat 2: mean delta +0.01852, P(delta>0)=0.812
```

Important metric deltas:

```text
Better for pronking:
  vx_err_mean             -0.015895
  score_progress          +0.021230
  score_yaw_tracking      +0.086275
  score_contact_slip      +0.088293
  score_impact            +0.031390
  impact_velocity_rms     -0.084038

Worse for pronking:
  score_power_efficiency  -0.016295
  mechanical_power_abs    +12.019821
  score_vertical_bounce   -0.394660
  score_action_smoothness -0.021172
  score_gait_stability    -0.018524
```

Interpretation:

```text
The confirmation run supports the smoke-test finding:
  sustained pronking from a trotting-context ramp state at vx=1.25 is better
  than sustained trotting under canonical_efficiency_v4_physical.

This is not a pure energy-efficiency result. Pronking uses more mechanical
power and has worse vertical-bounce/action-smoothness style metrics. It wins
because tracking/progress, yaw tracking, contact slip, and impact are better.

The core contradiction is therefore:
  sustained pronking is beneficial when evaluated as a multi-step gait option;
  one-step sampled pronking inside PPO receives negative advantage and is
  suppressed.

This points to a temporal abstraction problem. Gait selection should be trained
as a persistent decision over several high-level steps, with the rollout buffer
credit assigned to that persistent decision. Wrapper-level selector hold is not
sufficient because it caused sampled/executed action mismatch when the policy
still sampled every step.
```

Next implementation direction:

```text
Add an explicit decision-interval training mode:
  - policy samples the gait action once every K high-level steps;
  - the selected action is actually executed for those K steps;
  - reward is accumulated over the K-step option;
  - PPO stores one transition for that option decision;
  - no wrapper-level hidden hold is needed for selector credit.

This should be tested first on ramp-only, selector-only, no task id, no gait
reference, residuals fixed at zero.
```

## 2026-07-08: Decision-Interval Training Mode

Code update:

```text
scripts/train_high_level_oracle_ppo.py
```

New option:

```text
--decision-interval K
```

Meaning:

```text
The high-level policy samples one gait action once, then the environment
executes that same high-level action for K high-level environment steps.
Only after those K steps does the trainer add one PPO transition to the rollout
buffer.
```

Chinese interpretation:

```text
这不是告诉网络“斜坡必须用 pronk”。
这也不是加入任务编号、地形标签、步态审查表，或者新的步态先验。

它只改变训练中“一个步态决定要为多长时间的结果负责”。
```

Why this is needed:

```text
The paired same-state audit showed:
  sustained pronking from a trotting-context ramp state at vx=1.25 is better
  than sustained trotting.

But ordinary one-step PPO training showed:
  sampled pronking receives negative advantage and is quickly suppressed.

This means the useful effect of a gait choice may require sustained execution
over several high-level steps. If the policy samples a new gait every step,
the reward assigned to a single sampled step can miss the longer-term effect
of actually committing to that gait.
```

Difference from `selector_hold_steps`:

```text
selector_hold_steps:
  the wrapper may keep executing an old gait while PPO has already sampled a
  new gait. This created sampled/executed mismatch around 0.47-0.50 in prior
  diagnostics.

decision_interval:
  the policy samples once, executes that same action for K steps, and stores
  one training sample for that K-step decision. Sampled and executed gait stay
  aligned.
```

Expected advantage of this experiment:

```text
It tests whether the clean reward already contains a usable ramp/pronking
signal once the training time scale matches the physical effect time scale.

If this works:
  the issue was not primarily missing task information or missing gait priors;
  the issue was that one-step gait decisions were too short to receive the
  correct multi-step reward attribution.

If this still converges to all-trotting:
  then even multi-step decision attribution is not enough, and the ramp/pronking
  advantage is still too weak/noisy under this clean training setup.
```

Important constraint:

```text
When --decision-interval > 1, the script now requires:

--selector-hold-steps 0

This avoids reintroducing the hidden sampled/executed gait mismatch.
```

First intended diagnostic:

```text
ramp-only
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
selector only
continuous residuals fixed at zero
selector uses command speed + student latent + predicted clean physical state
decision_interval = 5
selector_hold_steps = 0
```

Smoke run completed:

```text
runs/high_level_oracle_gait/20260708_v4_ramp_only_decision_interval5_smoke_iter005
```

Setup:

```text
ramp-only
decision_interval = 5
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Key readout:

```text
5 iterations completed
checkpoints saved: high_level_000000.pt, high_level_000004.pt
decision_interval column = 5
physical_steps per iteration = 160
sampled_executed_gait_mismatch_rate = 0.000 for every iteration
```

Early metrics:

```text
iter 0:
  reward = 0.7527
  vx_err = 0.2744
  gait ratios = pronking 0.226, trotting 0.278, bounding 0.278, pacing 0.218

iter 4:
  reward = 0.7481
  vx_err = 0.2880
  gait ratios = pronking 0.274, trotting 0.271, bounding 0.291, pacing 0.164
```

Interpretation:

```text
This smoke test only validates the mechanics of the new training mode. It does
not yet prove whether decision-interval training can learn ramp/pronking.

The important success is that sampled and executed gait now stay aligned while
one sampled gait action controls five high-level environment steps. This removes
the earlier wrapper-hold credit mismatch and makes the next 50-iteration ramp
diagnostic meaningful.
```

50-iteration ramp diagnostic completed:

```text
runs/high_level_oracle_gait/20260708_v4_ramp_only_decision_interval5_iter050
```

Setup:

```text
ramp-only
decision_interval = 5
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Run status:

```text
50 iterations completed
checkpoints saved: high_level_000000.pt, high_level_000049.pt
sampled_executed_gait_mismatch_rate = 0.000 for all iterations
```

Training summary:

```text
first 5 iterations:
  reward = 0.7653
  vx_err = 0.2537
  gait ratios = pronking 0.271, trotting 0.292, bounding 0.246, pacing 0.191

last 10 iterations:
  reward = 0.8058
  vx_err = 0.2192
  gait ratios = pronking 0.100, trotting 0.805, bounding 0.085, pacing 0.011

iteration 49:
  reward = 0.8049
  vx_err = 0.2201
  gait ratios = pronking 0.072, trotting 0.823, bounding 0.099, pacing 0.006
```

Late-window sampled-action advantage:

```text
last 10 iterations, all speeds:
  pronking -0.0358
  trotting +0.0032
  bounding +0.0366
  pacing   -0.1725

last 10 iterations, low speed 0.5-1.0:
  pronking -0.0049
  trotting +0.0570
  bounding +0.1097
  pacing   -0.2103

last 10 iterations, mid speed 1.0-1.5:
  pronking -0.0817
  trotting -0.0738
  bounding -0.0577
  pacing   +0.0321

last 10 iterations, high speed 1.5-2.0:
  pronking -0.0409
  trotting -0.0123
  bounding -0.0531
  pacing   -0.3811
```

Interpretation:

```text
Decision-interval training fixes the sampled/executed mismatch, but it does not
make clean ramp-only training learn a pronking-dominant selector.

By the final 10 iterations, the policy again becomes strongly trotting-biased.
The sampled-action advantage does not support pronking in any speed bin. This
means the previous failure cannot be explained only by one-step action duration
or wrapper-hold credit mismatch.

The next required step is deterministic fixed-speed evaluation of checkpoint
high_level_000049.pt. If that evaluation is also all-trotting, the current clean
reward-only line has now failed three increasingly fair ramp diagnostics:

  1. ordinary one-step PPO,
  2. wrapper hold-step PPO,
  3. explicit decision-interval PPO.
```

Deterministic evaluation of decision-interval checkpoint completed:

```text
runs/high_level_oracle_gait/20260708_v4_ramp_only_decision_interval5_iter050/independent_eval/20260708_ramp_full_iter049
```

Note:

```text
The command used --full, so it evaluated all standard task/speed rows, not only
ramp. The ramp rows are the relevant rows for this diagnosis.
```

Ramp result:

```text
ramp vx=0.5:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.9028
  vx_err = 0.0764

ramp vx=1.0:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.8424
  vx_err = 0.1578

ramp vx=1.5:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.7702
  vx_err = 0.2821

ramp vx=2.0:
  gait ratios = trotting 1.000, pronking 0.000, bounding 0.000, pacing 0.000
  reward = 0.6838
  vx_err = 0.5030
```

Additional observation:

```text
The same checkpoint is also deterministic all-trotting on the other evaluated
task/speed rows. This is expected for a ramp-only trained clean selector because
no direct task id or gait-reference table was used, but it reinforces that this
policy has collapsed to a global trotting solution.
```

Interpretation:

```text
decision_interval=5 did not solve the clean ramp/pronking learning problem.
It fixed the sampled/executed gait mismatch, but the final policy still chooses
trotting at every ramp speed.

The most likely explanation is now narrower:
  either the useful pronking advantage requires a longer sustained horizon than
  5 high-level steps,
  or the PPO training objective still gives trotting a broader/easier
  optimization basin despite sustained pronking being better in the paired
  long-horizon audit.
```

Next diagnostic:

```text
Before trying another training run, measure paired pronking-vs-trotting reward
as a function of horizon length, e.g. 5, 10, 20, 50, 100 high-level steps.

Reason:
  The paired confirmation used a long sustained rollout and showed pronking is
  better. The failed decision-interval run used K=5. If pronking does not beat
  trotting over a 5-step horizon, then K=5 was simply too short to expose the
  advantage to PPO.
```

5-step paired horizon audit completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260708_ramp_mid_pronk_vs_trot_context_trot_horizon005
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 64
repeats = 3
steps = 5
warmup_steps = 0
delta = pronking - trotting
```

Result:

```text
weighted_metric_reward:
  pronking = 0.8541
  trotting = 0.8938
  delta = -0.0397
  median delta = -0.0344
  P(delta > 0) = 0.016

vx_err_mean:
  pronking = 0.1041
  trotting = 0.0689
  delta = +0.0352

score_progress:
  pronking = 0.9352
  trotting = 0.9680
  delta = -0.0328

score_orientation:
  pronking = 0.7411
  trotting = 0.8952
  delta = -0.1541

score_power_efficiency:
  pronking = 0.5017
  trotting = 0.5710
  delta = -0.0693

mechanical_power_abs:
  pronking = 204.17
  trotting = 165.30
  delta = +38.87

score_impact:
  pronking = 0.8975
  trotting = 0.8688
  delta = +0.0287
```

Interpretation:

```text
This explains why decision_interval=5 did not learn pronking.

At a 5-step decision horizon, switching from a trotting-context state to
pronking is not beneficial under the live v4 reward. Trotting is clearly better
over this short horizon because it has better progress, orientation, energy,
scuffing, and vertical-bounce related behavior. Pronking's impact/yaw benefits
are not enough to compensate.

Therefore the failed decision_interval=5 training is not surprising. It was
not yet giving PPO the long enough sustained execution window in which the
previous long-horizon paired audit showed pronking becomes better.

The current question is now quantitative:
  at what sustained horizon does pronking overtake trotting on ramp vx=1.25?
```

20-step paired horizon audit completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260708_ramp_mid_pronk_vs_trot_context_trot_horizon020
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 64
repeats = 3
steps = 20
warmup_steps = 0
delta = pronking - trotting
```

Result:

```text
weighted_metric_reward:
  pronking = 0.8768
  trotting = 0.8944
  delta = -0.0176
  median delta = -0.0175
  P(delta > 0) = 0.141

vx_err_mean:
  pronking = 0.0861
  trotting = 0.0671
  delta = +0.0189

score_progress:
  pronking = 0.9480
  trotting = 0.9691
  delta = -0.0211

score_orientation:
  pronking = 0.8455
  trotting = 0.9033
  delta = -0.0578

score_contact_slip:
  pronking = 0.8181
  trotting = 0.7569
  delta = +0.0612

score_power_efficiency:
  pronking = 0.5364
  trotting = 0.5787
  delta = -0.0423

score_impact:
  pronking = 0.8846
  trotting = 0.8684
  delta = +0.0162
```

Interpretation:

```text
The 20-step horizon is still not long enough for pronking to overtake trotting
when switching from a trotting-context state.

However, the reward gap has narrowed:
  5-step delta  = -0.0397
  20-step delta = -0.0176

This supports the transition-cost interpretation. Pronking begins to recover
some ground over a longer window, mostly through contact slip and impact, but
still loses to trotting on progress, orientation, power, and vertical-bounce
related behavior over the first 20 high-level steps.

Next horizon should be 50 steps with warmup_steps=0. This asks whether the
previous long-horizon pronking advantage only appears after a substantial
settling period.
```

50-step paired horizon audit completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260709_ramp_mid_pronk_vs_trot_context_trot_horizon050
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 64
repeats = 3
steps = 50
warmup_steps = 0
delta = pronking - trotting
```

Result:

```text
weighted_metric_reward:
  pronking = 0.8237
  trotting = 0.8120
  delta = +0.0117
  median delta = +0.0069
  P(delta > 0) = 0.557

vx_err_mean:
  pronking = 0.1920
  trotting = 0.2054
  delta = -0.0134

score_progress:
  pronking = 0.8466
  trotting = 0.8289
  delta = +0.0177

score_yaw_tracking:
  pronking = 0.7312
  trotting = 0.6571
  delta = +0.0742

score_contact_slip:
  pronking = 0.8151
  trotting = 0.7388
  delta = +0.0763

score_impact:
  pronking = 0.8941
  trotting = 0.8657
  delta = +0.0284

score_power_efficiency:
  pronking = 0.5294
  trotting = 0.5477
  delta = -0.0183

mechanical_power_abs:
  pronking = 197.15
  trotting = 184.07
  delta = +13.08
```

Horizon trend so far:

```text
5-step delta   = -0.0397, P(pronk > trot) = 0.016
20-step delta  = -0.0176, P(pronk > trot) = 0.141
50-step delta  = +0.0117, P(pronk > trot) = 0.557
300-step delta = +0.0178, P(pronk > trot) = 0.760  (previous confirmation)
```

Interpretation:

```text
The transition point is real. From a trotting-context ramp state at vx=1.25,
pronking is worse over 5 and 20 high-level steps, but becomes better by 50
steps.

This explains why decision_interval=5 failed: the training horizon was still
inside the transition-cost regime. It also suggests that ordinary one-step PPO
and short decision-interval PPO both punish pronking before its sustained
benefits appear.

However, the 50-step advantage is still noisy:
  P(delta > 0) = 0.557
  delta_std = 0.0542

So the signal is not yet a strong deterministic preference. The previous
300-step confirmation was stronger, which means the pronking advantage keeps
becoming clearer over a longer sustained horizon.
```

Next diagnostic:

```text
Run the same paired audit with steps=100, warmup_steps=0.

Purpose:
  Confirm whether the reward advantage and positive-rate continue moving toward
  the 300-step result, and estimate whether a practical training decision
  interval would need to be closer to 50, 100, or even longer.
```

100-step paired horizon audit completed:

```text
runs/high_level_oracle_gait/paired_gait_live_reward_audit/20260709_ramp_mid_pronk_vs_trot_context_trot_horizon100
```

Setup:

```text
ramp_up_trot_robustness vx=1.25
gait_a = pronking
gait_b = trotting
context_gait = trotting
context_steps = 20
num_envs = 64
repeats = 3
steps = 100
warmup_steps = 0
delta = pronking - trotting
```

Result:

```text
weighted_metric_reward:
  pronking = 0.8199
  trotting = 0.8085
  delta = +0.0114
  median delta = +0.0147
  P(delta > 0) = 0.672

vx_err_mean:
  pronking = 0.2011
  trotting = 0.2106
  delta = -0.0095

score_progress:
  pronking = 0.8382
  trotting = 0.8233
  delta = +0.0149

score_yaw_tracking:
  pronking = 0.7344
  trotting = 0.6547
  delta = +0.0796

score_contact_slip:
  pronking = 0.8171
  trotting = 0.7352
  delta = +0.0819

score_impact:
  pronking = 0.8960
  trotting = 0.8667
  delta = +0.0293

score_power_efficiency:
  pronking = 0.5254
  trotting = 0.5458
  delta = -0.0204

mechanical_power_abs:
  pronking = 201.66
  trotting = 184.46
  delta = +17.19
```

Updated horizon trend:

```text
5-step delta   = -0.0397, P(pronk > trot) = 0.016
20-step delta  = -0.0176, P(pronk > trot) = 0.141
50-step delta  = +0.0117, P(pronk > trot) = 0.557
100-step delta = +0.0114, P(pronk > trot) = 0.672
300-step delta = +0.0178, P(pronk > trot) = 0.760
```

Interpretation:

```text
The 100-step result confirms the horizon-dependent story.

The average reward advantage is similar to the 50-step result, but the positive
rate improves from 0.557 to 0.672. This means pronking is not becoming much more
profitable on average between 50 and 100 steps, but the advantage is becoming
more reliable across environments.

The gait preference is therefore a long-horizon effect:
  short horizon: trotting clearly better;
  medium horizon: pronking starts to overtake on average;
  long horizon: pronking advantage becomes more consistent.

This is a poor match for ordinary one-step PPO and still difficult for short
decision-interval PPO. A practical training interval may need to be much longer
than 5, likely at least 50, and possibly closer to 100 if we want the selector
to see a stable positive signal.
```

Next decision:

```text
Do not jump straight to a very long decision-interval training run yet.

First update the training code so decision-interval mode can keep the same
approximate number of physical simulator steps while changing only the decision
interval. Otherwise increasing decision_interval from 5 to 50 or 100 also
changes total rollout horizon, batch size, and compute cost in a way that makes
results harder to interpret.
```

Code update added:

```text
scripts/train_high_level_oracle_ppo.py
```

New option:

```text
--num-physical-steps N
```

Meaning:

```text
When this option is set, the script keeps N high-level environment steps per
training iteration and derives the number of PPO decision samples as:

decision_steps = num_physical_steps / decision_interval
```

Reason:

```text
Without this option:
  --num-steps controls the number of policy decisions.
  Increasing decision_interval from 5 to 50 would also multiply the true
  simulator steps per iteration by 10.

With this option:
  the true simulator steps per iteration stay fixed.
  Changing decision_interval mainly changes how long each gait decision is
  held responsible for future reward.
```

This keeps the comparison cleaner:

```text
decision_interval=5:
  many short decisions per iteration

decision_interval=50:
  fewer long decisions per iteration

both can use the same total physical simulator steps per iteration if
--num-physical-steps is held fixed.
```

Validation:

```text
python3 -m py_compile scripts/train_high_level_oracle_ppo.py
```

Decision-interval 50 smoke run completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_only_decision_interval50_phys800_smoke_iter005
```

Setup:

```text
ramp-only
decision_interval = 50
num_physical_steps = 800
rollout_decision_steps = 16
rollout_physical_steps = 800
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Run status:

```text
5 iterations completed
checkpoints saved: high_level_000000.pt, high_level_000004.pt
sampled_executed_gait_mismatch_rate = 0.000 for all iterations
```

Early readout:

```text
iter 0:
  reward = 0.7972
  vx_err = 0.2211
  gait ratios = pronking 0.233, trotting 0.246, bounding 0.267, pacing 0.254

iter 4:
  reward = 0.8112
  vx_err = 0.2095
  gait ratios = pronking 0.344, trotting 0.345, bounding 0.146, pacing 0.166
```

Interpretation:

```text
The new fixed-physical-step decision-interval code path is working:
  - decision_interval is correctly set to 50;
  - each iteration uses 800 high-level environment steps;
  - PPO stores 16 decision samples per environment per iteration;
  - sampled and executed gait remain aligned.

This smoke run is still too short to judge convergence. The next useful test is
a 50-iteration ramp-only run with the same decision interval and physical-step
budget.
```

Decision-interval 50 formal ramp-only run completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_only_decision_interval50_phys800_iter050
```

Setup:

```text
ramp-only
decision_interval = 50
num_physical_steps = 800
rollout_decision_steps = 16
rollout_physical_steps = 800
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Run status:

```text
50 iterations completed
checkpoints saved: high_level_000000.pt, high_level_000049.pt
sampled_executed_gait_mismatch_rate = 0.000 for all iterations
```

Training-log summary:

```text
first 5 iterations:
  reward = 0.8029
  vx_err = 0.2188
  gait ratios = pronking 0.303, trotting 0.284, bounding 0.214, pacing 0.199
  gait_switch_rate = 0.742

iterations 20-29:
  reward = 0.8281
  vx_err = 0.1975
  gait ratios = pronking 0.705, trotting 0.284, bounding 0.004, pacing 0.006
  gait_switch_rate = 0.422

last 10 iterations:
  reward = 0.8312
  vx_err = 0.1937
  gait ratios = pronking 0.873, trotting 0.126, bounding 0.001, pacing 0.000
  gait_switch_rate = 0.216

final iteration:
  reward = 0.8295
  vx_err = 0.1974
  gait ratios = pronking 0.938, trotting 0.062, bounding 0.000, pacing 0.000
  gait_switch_rate = 0.110
```

Interpretation:

```text
This is the first clean no-reference ramp-only training run that does not
collapse to deterministic all-trotting in the training log. Increasing the
explicit decision interval from 5 to 50 changes the training direction: the
policy gradually moves from mixed exploration to a mostly-pronking ramp policy.

This matches the paired same-state horizon audits:
  - switching to pronking from a trotting-context state is worse over 5 and 20
    high-level steps;
  - pronking starts to overtake around 50 steps;
  - the advantage becomes more reliable at longer horizons.

Therefore the earlier clean ramp-only failures were not just caused by missing
terrain information or by the reward lacking a pronking signal. A major issue
was temporal credit assignment: short one-step or 5-step gait decisions were
rewarded before pronking had enough time to realize its sustained-ramp
advantage.

This is still not a final policy result. Training-time gait ratios are sampled
under the training distribution. The next required step is fixed-speed
independent evaluation of checkpoint high_level_000049.pt on ramp speeds
0.5, 1.0, 1.5, and 2.0 using the same canonical reward profile.
```

Fixed-speed independent evaluation of the decision-interval 50 checkpoint
completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_only_decision_interval50_phys800_iter050/independent_eval/20260709_ramp_full_iter049
```

Result:

```text
ramp vx=0.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.9032
  vx_err = 0.0774

ramp vx=1.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.8568
  vx_err = 0.1411

ramp vx=1.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.7837
  vx_err = 0.2759

ramp vx=2.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.6757
  vx_err = 0.5167
```

Comparison:

```text
average reward of this decision-interval 50 policy:
  0.8048

previous clean all-trotting ramp-only policy:
  0.7985

average improvement:
  +0.0063

dense fixed-gait best default-template average:
  0.8073

gap to dense fixed-gait best:
  -0.0025
```

Interpretation:

```text
The fixed-speed evaluation confirms that the decision-interval 50 policy is not
only stochastically sampling pronking during training. Deterministic evaluation
selects pronking at every tested ramp speed.

This is a meaningful clean result:
  no direct task id;
  no gait-reference supervision table;
  no per-terrain reward;
  continuous residuals fixed at zero;
  unified canonical_efficiency_v4_physical reward.

The result supports the temporal-credit explanation. When the high-level gait
decision is held long enough for the selected gait's sustained effect to matter,
the same clean reward can train a ramp/pronking policy.

Remaining limitation:
  the learned policy overgeneralizes pronking to vx=2.0. The dense fixed-gait
  audit showed trotting is better at vx=2.0, so the policy has learned a
  ramp-level pronking solution rather than the full speed-conditioned optimum.

Next question:
  whether adding more speed-specific evidence or using an intermediate/variable
  decision interval can preserve pronking at ramp mid-speeds while allowing
  trotting to reappear at the high-speed edge.
```

Ramp+flat curriculum continuation from the decision-interval 50 ramp checkpoint
completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_from_ramp_iter050
```

Setup:

```text
initialized from:
  runs/high_level_oracle_gait/20260709_v4_ramp_only_decision_interval50_phys800_iter050/checkpoints/high_level_000049.pt

included tasks:
  ramp_up_trot_robustness
  flat_trot_efficiency

decision_interval = 50
num_physical_steps = 800
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Training-log summary:

```text
first 5 iterations:
  reward = 0.8368
  vx_err = 0.1909
  all gait ratios = pronking 0.950, trotting 0.050
  flat ratios = pronking 0.944, trotting 0.056
  ramp ratios = pronking 0.955, trotting 0.045

iterations 20-29:
  reward = 0.8380
  vx_err = 0.1890
  all gait ratios = pronking 0.997, trotting 0.002
  flat ratios = pronking 0.996, trotting 0.004
  ramp ratios = pronking 0.999, trotting 0.001

last 10 iterations:
  reward = 0.8379
  vx_err = 0.1891
  all gait ratios = pronking 1.000, trotting 0.000
  flat ratios = pronking 1.000, trotting 0.000
  ramp ratios = pronking 1.000, trotting 0.000
  gait_switch_rate = 0.000
```

Interpretation:

```text
The ramp+flat continuation did not produce the desired condition split:
  ramp -> pronking
  flat -> trotting

Instead, the ramp-pronking solution generalized to flat and became an all-
pronking policy for both included tasks.

This is an important negative result. The decision-interval 50 mechanism solves
one problem, namely that pronking needs enough sustained execution time before
its ramp advantage appears. It does not by itself solve the separate problem of
condition-dependent gait selection across overlapping terrain distributions.

The current clean curriculum line therefore has two demonstrated failure modes:
  - short decision horizon: all-trotting basin dominates even on ramp;
  - long decision horizon initialized from ramp: pronking can dominate and
    spread to flat.

Before adding more terrains, the next required check is fixed-speed independent
evaluation on flat+ramp for checkpoint high_level_000049.pt. If deterministic
evaluation confirms all-pronking on flat and ramp, then the next design question
is how to preserve the useful long-horizon gait effect while still giving the
selector enough contrast to recover flat/trot.
```

Fixed-speed independent evaluation of the ramp+flat continuation completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_from_ramp_iter050/independent_eval/20260709_flat_ramp_iter049
```

Result:

```text
flat vx=0.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.9149
  vx_err = 0.0602

flat vx=1.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.8661
  vx_err = 0.1346

flat vx=1.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.7914
  vx_err = 0.2756

flat vx=2.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.6943
  vx_err = 0.4887

ramp vx=0.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.9029
  vx_err = 0.0778

ramp vx=1.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.8556
  vx_err = 0.1430

ramp vx=1.5:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.7832
  vx_err = 0.2783

ramp vx=2.0:
  gait ratios = pronking 1.000, trotting 0.000, bounding 0.000, pacing 0.000
  reward = 0.6746
  vx_err = 0.5230
```

Aggregate:

```text
flat average reward = 0.8167
flat average vx_err = 0.2398

ramp average reward = 0.8041
ramp average vx_err = 0.2555
```

Interpretation:

```text
The fixed-speed evaluation confirms the training-log diagnosis: the ramp+flat
continuation is deterministically all-pronking on both flat and ramp.

This means the failure is not only stochastic exploration during training. Once
the ramp-pronking solution is learned with a long decision interval, it becomes
a global attractor and also covers flat.

Therefore the next issue is no longer whether long decision intervals can reveal
pronking. They can. The issue is whether the unified reward and training setup
provide enough contrast for flat/trotting to compete against a sustained
pronking policy.

Before any further curriculum expansion, run a flat fixed-gait live reward
check comparing trotting and pronking under the same canonical reward and
default continuous parameters. This asks whether flat/pronking is actually being
rewarded nearly as well as flat/trotting, or whether the curriculum has pushed
the policy into a suboptimal all-pronking basin.
```

Flat fixed-gait trot-vs-pronk live-reward check completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260709_v4_flat_default_trot_pronk_check
```

Result:

```text
flat vx=0.5:
  trotting reward = 0.9163
  pronking reward = 0.9150
  trotting - pronking = +0.0013

flat vx=1.0:
  trotting reward = 0.8732
  pronking reward = 0.8664
  trotting - pronking = +0.0068

flat vx=1.5:
  trotting reward = 0.8113
  pronking reward = 0.7925
  trotting - pronking = +0.0189

flat vx=2.0:
  trotting reward = 0.7353
  pronking reward = 0.6945
  trotting - pronking = +0.0407

average:
  trotting reward = 0.8340
  pronking reward = 0.8171
  trotting - pronking = +0.0169
```

Interpretation:

```text
The flat all-pronking behavior is not supported by the fixed-gait reward audit.
Under the same canonical_efficiency_v4_physical reward and default continuous
parameters, trotting beats pronking at every tested flat speed. The gap is tiny
at vx=0.5, but grows clearly at higher speed.

The ramp+flat continuation's flat average reward was 0.8167, almost identical
to fixed flat-pronking 0.8171 and clearly below fixed flat-trotting 0.8340.

Therefore the ramp+flat curriculum did push the policy into a suboptimal
all-pronking basin on flat. This is not simply a reward-definition issue where
the reward secretly prefers pronking on flat.

Current diagnosis:
  - short decision horizon favors the all-trotting basin;
  - long decision horizon lets ramp/pronking appear;
  - but long decision horizon plus ramp-first curriculum can make pronking
    become a global basin and overwrite flat/trotting.

The next design needs a way to keep the long-horizon benefit while restoring
condition contrast. Candidate directions:
  1. train flat and ramp together from scratch with decision_interval=50 rather
     than starting from a ramp-pronking checkpoint;
  2. use balanced sampling or replay during curriculum so flat/trotting is never
     absent while ramp/pronking is learned;
  3. consider a shorter or variable decision interval if K=50 is too coarse for
     recovering speed/terrain-specific switching.
```

Ramp+flat decision-interval 50 training from scratch completed:

```text
runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_scratch_iter050
```

Setup:

```text
included tasks:
  ramp_up_trot_robustness
  flat_trot_efficiency

init_checkpoint = none
decision_interval = 50
num_physical_steps = 800
selector_hold_steps = 0
selector-only
continuous residuals fixed at zero
no direct task id
no gait-reference supervision table
canonical_efficiency_v4_physical
```

Training-log summary:

```text
first 5 iterations:
  reward = 0.8109
  vx_err = 0.2048
  all gait ratios = pronking 0.253, trotting 0.244, bounding 0.294, pacing 0.210
  flat ratios = pronking 0.245, trotting 0.239
  ramp ratios = pronking 0.262, trotting 0.249

iterations 20-29:
  reward = 0.8368
  vx_err = 0.1893
  all gait ratios = pronking 0.925, trotting 0.057, bounding 0.014, pacing 0.003
  flat ratios = pronking 0.921, trotting 0.059
  ramp ratios = pronking 0.929, trotting 0.055

last 10 iterations:
  reward = 0.8373
  vx_err = 0.1899
  all gait ratios = pronking 0.989, trotting 0.008, bounding 0.002, pacing 0.001
  flat ratios = pronking 0.981, trotting 0.013
  ramp ratios = pronking 0.996, trotting 0.003
```

Interpretation:

```text
Training flat+ramp together from scratch did not recover the intended contrast.
It also converged to an almost all-pronking policy.

This rejects the narrower explanation that the previous all-pronking result was
only caused by initializing from a ramp-pronking checkpoint. With
decision_interval=50, pronking becomes a strong attractor even when flat is
present from the start.

Combined with the flat fixed-gait audit, this is an important mismatch:
  fixed-gait reward says flat/trotting is better than flat/pronking;
  PPO with long decision interval nevertheless moves toward flat/pronking.

The next required check is fixed-speed independent evaluation of the scratch
checkpoint high_level_000049.pt on flat+ramp. If deterministic evaluation also
confirms all-pronking, then the next design step should focus on why the PPO
training signal under long decision intervals does not preserve the flat/trot
advantage seen in fixed-gait audit.
```

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
  - selector-only policy/evaluation functions,
  - optional predicted-physical-state input to the gait selector.

- `go2_gym/envs/wrappers/high_level_gait_wrapper.py`
  - 9D high-level action wrapper,
  - gait template and continuous parameter mapping,
  - live high-level reward and metric scores.

- `go2_gym/envs/wrappers/high_level_reward_metrics.py`
  - shared canonical high-level reward score formulas,
  - used by online wrapper and offline consistency checks.

- `scripts/evaluate_high_level_policy_by_task.py`
  - deterministic independent checkpoint evaluation,
  - one task/speed at a time,
  - reports executed gait ratios and task metrics.

- `scripts/evaluate_fixed_gait_live_reward.py`
  - fixed-gait audit script,
  - one task/speed/gait at a time,
  - directly measures the current live training reward for fixed gait actions.

- `scripts/check_high_level_reward_consistency.py`
  - same-trajectory online/offline reward consistency check,
  - mandatory before rerunning expensive fair-grid audits or PPO with a new
    unified reward candidate.

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

### Reference-supervised selector runs

The runs using `selector_targets.csv` are not the same as pure reward training.
They use a fair-audit-derived reference probability for the gait selector during
training.

This is useful as a diagnostic because it answers:

```text
If the gait selector is given a clear reference signal, can the architecture use
RMA/proprioceptive information and command speed to produce condition-dependent
gait choices?
```

It should not be used to claim:

```text
The unified physical reward alone caused the gait families to emerge.
```

Current interpretation:

```text
The reference-supervised selector baseline is an upper-bound / diagnostic model.
It is appropriate for checking the information pathway, visual behavior, and
whether the policy can express gait differentiation at all.

It is not the clean deployment or sim-to-real generalization claim, because
training required a gait-reference table derived from audited conditions.
```

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

Current immediate step as of 2026-07-04:

```text
Separate the project into two tracks:

1. Clean generalization track:
   no direct task id
   no gait-reference supervision table
   unified physical reward
   proprioceptive history / RMA only

2. Diagnostic / upper-bound track:
   no direct task id at inference
   fair-audit-derived gait-reference supervision during training
   selector-only/default continuous parameters

The current strong pronking/trotting differentiation belongs to track 2, not
track 1.
```

Current diagnostic reference input:

```text
runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv
```

Current profile:

```text
canonical_efficiency_v4_physical
```

Current interpretation:

```text
Pure v4 physical-reward training did not naturally create clear gait-family
separation. The reference-supervised selector diagnostics show that the network
can express condition-dependent gait selection when the gait selector is given
a fair-audit-derived reference signal during training.

This is useful, but it is not the main claim that unified physical reward
naturally produces gait switching.

Current reference-supervised diagnostic baseline:
  runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/checkpoints/high_level_000099.pt

It has:
  no direct task id at inference
  gait selector input = student RMA latent + command speed
  continuous residuals fixed at zero/default WTW parameters
  reference top-gait match = 11 / 17 on the fixed training-range evaluation
  high-confidence match = 7 / 7
  repeat-stable top-gait decisions
  smooth intermediate-speed behavior
  visible pronking/trotting condition response on flat/ramp/rough/push

But it should be described as:
  diagnostic / upper-bound evidence that the architecture can use condition
  information when given a gait-selection training signal.

It should not be described as:
  evidence that the robot independently discovered gait rules from the unified
  physical reward alone.
```

Near-term work:

```text
Do not further optimize the reference-supervised baseline as the final claim.
Use it only as a diagnostic and visualization reference.

The next clean experiment should remove the gait-reference supervision table
and evaluate whether unified physical reward plus proprioceptive/RMA evidence
can recover useful locomotion adaptation without explicit gait-selection labels.

If clean training does not produce visible gait switching but improves tracking,
stability, energy, or OOD robustness, report that honestly: the unified physical
objective may favor mostly trotting plus state-dependent recovery behavior
rather than broad discrete gait-family switching.
```

### Clean physical-state latent training

Implemented support:

```text
--privileged-obs-mode clean_physics
```

This keeps the privileged observation shape unchanged but zeros:

```text
push_active
push_axis
```

Reason:

```text
These two fields are too close to task labels. Keeping them in the teacher
target makes the RMA path less clean for generalization. The policy should infer
push/recovery-like state from body/contact history, not from a simulation flag
that says the push condition is active.
```

Implemented support:

```text
--physical-state-coef <value>
```

This adds a training loss from the student RMA latent to the cleaned generic
physical privileged observation.

Important interpretation:

```text
This is not gait supervision.
It does not say "flat should trot" or "ramp should pronk".
It only asks the student latent to encode generic physical state, such as
terrain height statistics, slope proxy, friction/mass/COM changes, body height,
pitch proxy, and roll proxy.
```

Recommended next clean experiment:

```text
no direct task id
no selector_targets
no gait-reference supervision table
reward_profile = canonical_efficiency_v4_physical
selector_latent_cmd_only = true
selector_only = true
continuous residuals fixed at zero
privileged_obs_mode = clean_physics
physical_state_coef > 0
```

Pass/fail interpretation:

```text
If this improves condition response without gait-reference supervision, then
the bottleneck was partly the proprioception/RMA information pathway.

If it remains mostly trotting but improves tracking/stability/recovery, that is
still a valid unified-reward result.

If it neither improves gait behavior nor physical metrics, then the next issue
is not solved by adding more priors; we need to inspect whether the physical
state target is observable from the history, whether the reward is too weak for
selector differentiation, or whether the frozen low-level/templates do not
provide useful alternatives under this objective.
```

Smoke run:

```text
runs/high_level_oracle_gait/20260704_v4_clean_physics_notask_selectoronly_smoke_iter005
```

Setup:

```text
no direct task id
no selector_targets
selector_latent_cmd_only = true
selector_only = true
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
iterations = 5
```

Result:

```text
selector_target_training = false
style_reward_scale = 0.0, so task-map target_gait labels are not direct rewards
physical_state_loss: 0.1342 -> 0.0835
adaptation_loss:     0.0494 -> 0.0095
z_error:             0.0193 -> 0.0084
```

Interpretation:

```text
The clean physical-state auxiliary path is functional. It can train without a
gait-reference table and without direct task labels. The five-iteration run is
only a smoke test; gait ratios are not meaningful yet.
```

100-iteration clean run:

```text
runs/high_level_oracle_gait/20260704_v4_clean_physics_notask_zcmd_selectoronly_physcoef010_iter100
```

Setup:

```text
no direct task id
no selector_targets / no gait-reference supervision table
selector_latent_cmd_only = true
selector_only = true
residual_train_dims = none
privileged_obs_mode = clean_physics
physical_state_coef = 0.1
```

Training-log result:

```text
reward:              0.6518 -> 0.7254
vx_err:              0.4370 -> 0.3748
physical_state_loss: 0.1444 -> 0.0348
adaptation_loss:     0.0483 -> 0.0024
z_error:             0.0230 -> 0.0024
gait_switch_rate:    0.2258 -> 0.1245
```

Final training-time gait ratios:

```text
pronking = 0.179
trotting = 0.740
bounding = 0.063
pacing = 0.018
```

Interpretation:

```text
The clean physical-state auxiliary is learning: the student latent predicts the
cleaned physical state much better by the end of training, and speed tracking
also improves in the training log.

However, without the gait-reference table the gait choice still trends strongly
toward trotting. This supports the current hypothesis that a unified physical
reward plus better physical-state encoding may improve locomotion metrics, but
does not by itself guarantee visible multi-gait family differentiation.

This is not a code failure. It is the clean result we needed to test. The next
step is fixed 17-row independent evaluation of the final checkpoint, because
training-time ratios are not final evidence.
```

Independent fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260704_v4_clean_physics_notask_zcmd_selectoronly_physcoef010_iter100/independent_eval/20260704_training_range_iter099
```

Result:

```text
avg reward = 0.8261
avg vx_err = 0.3475
done_rate = 0.0208
gait_switch_rate = 0.0000
top gait counts = trotting 17 / 17
```

Comparison with the reference-supervised diagnostic baseline:

```text
clean physical-state run:
  avg reward = 0.8261
  avg vx_err = 0.3475
  top gait counts = trotting 17 / 17

reference-supervised diagnostic baseline:
  avg reward = 0.8195
  avg vx_err = 0.3533
  top gait counts = pronking 6 / 17, trotting 11 / 17
```

Interpretation:

```text
The clean no-reference run is not worse as a locomotion policy under the current
unified reward. It is slightly better on average reward and speed error than
the reference-supervised diagnostic baseline.

But it has no visible gait-family differentiation: every fixed evaluation row
selects trotting.

This is strong evidence that, under canonical_efficiency_v4_physical and default
continuous parameters, the unified physical objective naturally supports a
stable all-trotting policy more than multi-gait switching. Better physical-state
encoding helps training health, but does not by itself create gait
differentiation.
```

Information-path probe:

```text
runs/high_level_oracle_gait/info_path_probe/20260704_clean_physics_no_reference_iter099/analysis
```

Probe results:

```text
history -> task test_acc = 0.597
history -> speed test_acc = 0.847
z_student -> task test_acc = 0.565
z_student -> speed test_acc = 0.391
z_teacher -> task test_acc = 0.934
z_teacher -> speed test_acc = 0.332
```

Latent-swap sensitivity:

```text
student vs teacher mean gait-prob diff = 0.0023
student vs zero latent mean gait-prob diff = 0.0065
student vs shuffled latent mean gait-prob diff = 0.0017
```

Interpretation:

```text
The raw proprioceptive history contains usable speed information and partial
condition information. The student latent contains weaker condition information
and little speed information.

Most importantly, the gait-selection output barely responds when the latent is
replaced, zeroed, or shuffled. Therefore the clean all-trot policy is not
because the history contains no information. It is because the learned gait
selector is not using that information strongly enough, and the unified reward
does not punish the all-trot solution.

The next clean direction should not add gait labels. It should test whether a
different information pathway can make the gait selector use physical-state
features more directly while still leaving the final gait choice to the unified
physical reward.
```

## Historical Immediate Step: Offline Unified Re-Scoring

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

Live unified-efficiency audit completed:

```text
runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260615_unified_efficiency
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

Readout:

```text
fixed-gait live audit:
  top counts: pronk 10, trot 3, pace 1
  mean top-vs-second margin: 0.013
  clear margins >= 0.02: 3 / 14

fair-grid offline re-score using the same live proxy, with each gait allowed
to use its own best continuous parameters:
  unified_efficiency: pronk 15, trot 1, pace 1
  unified_balanced: pronk 15, trot 1, pace 1
```

Interpretation:

```text
The current live unified reward proxy is not suitable for PPO training. It
collapses toward pronking before learning starts. The main positive terms for
pronk are energy and slip; progress often pushes against pronk, but not enough.
The live wrapper does not yet expose explicit impact/scuff/contact-safety terms
that were important in the fair-grid analysis.
```

Decision:

```text
Do not run the short PPO smoke run with the current unified_efficiency proxy.
Do not assume the failure is explained only by missing impact/scuff terms.
There are at least three mismatches to resolve:

1. fixed-gait live audit used continuous residual = 0, while fair audit uses
   per-gait continuous-parameter search;
2. offline efficiency included impact/scuff scores, while the live proxy did
   not;
3. same-named online/offline metrics may still differ in raw quantity,
   normalization, averaging, warmup handling, reset handling, or done handling.
```

Safety guard:

```text
scripts/train_high_level_oracle_ppo.py refuses to train with diagnostic-only
reward profiles by default. Only profiles marked validated_for_training are
accepted without a deliberate diagnostic override.
```

Corrected next route:

```text
1. Metric-definition alignment
   Define one canonical unified reward from primitive quantities:
   progress, yaw, orientation, lateral, slip, energy, impact, scuff/contact
   safety, smoothness, survival, clearance, and boundary.

2. Same-trajectory online/offline consistency test
   This is the mandatory run-before-expensive-runs check. It means:
     - choose a small set of task/speed/gait/continuous-parameter combinations;
     - run one rollout and save primitive quantities such as base velocity,
       angular velocity, projected gravity, foot positions/velocities, terrain
       height, contact force, torque, joint velocity, action, done/reset flags;
     - compute scores and total reward inside HighLevelGaitWrapper online;
     - recompute the same scores offline from the saved trajectory;
     - compare each metric and total reward on the same trajectory.
   Passing criterion: online/offline per-metric and total reward differences
   are near zero up to numerical/aggregation tolerance.

3. Rerun live fair continuous-parameter search
   Once the objective changes, old fair-grid optima are not final evidence.
   Re-run the equal-budget live fair search under the corrected objective, with
   best rows selected by `weighted_metric_reward_mean`.

4. Live audit after consistency passes
   Run fixed-gait only as a quick smoke diagnostic, then run fair-grid/live
   audit where each gait gets equal continuous-parameter budget.

5. PPO only after the above
   Training profiles must be marked validated_for_training before the trainer
   accepts them by default.
```

Implementation started on 2026-06-16:

```text
go2_gym/envs/wrappers/high_level_reward_metrics.py
  shared canonical score formulas and weighted reward computation

go2_gym/envs/wrappers/high_level_gait_wrapper.py
  online wrapper now calls the shared score table
  records optional same-trajectory reward primitives
  exposes explicit impact and scuffing/contact-safety scores

scripts/check_high_level_reward_consistency.py
  small mandatory consistency check:
    online HighLevelGaitWrapper reward terms
    vs offline recomputation from the same recorded trajectory primitives

scripts/rescore_fair_grid_live_profiles.py
  re-score a completed fair gait grid with multiple live reward profiles
  using saved `score_<metric>` columns; no IsaacGym rollout is performed

scripts/train_high_level_oracle_ppo.py
  reward-profile status guard:
    task_focus_v4 -> validated_for_training
    unified_efficiency -> diagnostic_only_incomplete_proxy
    unified_balanced -> diagnostic_only_incomplete_proxy
    canonical_efficiency_candidate -> diagnostic_only_unvalidated_candidate
    canonical_balanced_candidate -> diagnostic_only_unvalidated_candidate
```

First server command:

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

If this check fails:

```text
Do not run fair-grid search, live audit, or PPO. Inspect
reward_consistency.csv and fix whichever metric differs between online and
offline computation.
```

Implementation note:

```text
The consistency script runs one task/speed/gait/residual case per child
process by default. This is intentional. IsaacGym and the global Cfg object are
not safe to repeatedly reinitialize for many different envs in one long Python
process; doing so can leave a config section as a dict and cause errors such as
`AttributeError: 'dict' object has no attribute 'command_curriculum'`.
```

If this check passes:

```text
Next run a larger consistency check with more task/speed/gait/residual
combinations, then rerun or revalidate fair continuous-parameter search under
the canonical candidate objective.
```

First consistency-check result:

```text
runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate

reward_profile: canonical_efficiency_candidate
cases: 16
  flat_trot_efficiency:1.0
  stepping_stones_easy_bound_highspeed:2.0
  pronking/trotting/bounding/pacing
  residual sets: zero, high_clearance

max_abs_error: 0
passed: True
```

Readout:

```text
The online wrapper reward terms and offline recomputation from recorded reward
primitives are exactly consistent for the checked cases. This confirms the
shared formula/aggregation path. It does not yet validate the full fair-search
dataset or prove that canonical_efficiency_candidate is a good training reward.
```

Recommended next consistency command:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/check_high_level_reward_consistency.py \
  --reward-profile canonical_efficiency_candidate \
  --eval flat_trot_efficiency:0.5,flat_trot_efficiency:1.5,ramp_up_trot_robustness:1.0,rough_slope_trot_robustness:1.0,push_lateral_pace_recovery:1.5,stepping_stones_easy_bound_highspeed:2.0 \
  --gaits pronking,trotting,bounding,pacing \
  --residual-sets 'zero=0,0,0,0,0;high_clearance=0,0,1,1,0;wide_low=0,0,-1,1,0;fast_narrow=1,0,0,-1,0' \
  --num-envs 4 \
  --steps 20 \
  --warmup-steps 2 \
  --output-dir runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate_broad
```

Broad consistency-check result:

```text
runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate_broad

reward_profile: canonical_efficiency_candidate
cases: 96
  task/speed points:
    flat_trot_efficiency:0.5
    flat_trot_efficiency:1.5
    ramp_up_trot_robustness:1.0
    rough_slope_trot_robustness:1.0
    push_lateral_pace_recovery:1.5
    stepping_stones_easy_bound_highspeed:2.0
  gaits:
    pronking, trotting, bounding, pacing
  residual sets:
    zero, high_clearance, wide_low, fast_narrow

metrics per case: 39
max_abs_error: 0
passed: True
```

Readout:

```text
The canonical reward formula/aggregation path is consistent for the broader
representative coverage. This clears the next gate: live fair continuous-
parameter search under canonical_efficiency_candidate.
```

Decision:

```text
Do not replace the canonical live fair search with an old-grid raw-metric
re-score. The old 20260614 fair-grid raw metrics remain useful for sanity
checks and rough expectation setting, but they are not final evidence for the
corrected objective because:
  - canonical_efficiency_candidate added explicit online impact/scuff scores;
  - the live wrapper computes scores per step before aggregation;
  - the final selection score is now weighted_metric_reward_mean, not the old
    neutral_score.

Therefore, if the goal is a defensible statement about each gait's best
continuous parameters under canonical_efficiency_candidate, the live fair search
is necessary despite the runtime cost.
```

Reward-profile reuse rule:

```text
One completed live fair search can be reused to compare multiple unified reward
standards if, and only if, those standards are different weightings over the
same recorded canonical score columns:

  score_progress
  score_yaw_tracking
  score_orientation
  score_pitch_rate / score_roll_rate / score_yaw_rate
  score_lateral_drift
  score_vertical_bounce
  score_slip
  score_energy
  score_impact
  score_scuffing
  score_clearance
  score_gait_stability
  score_action_smoothness
  score_action_magnitude
  score_action_boundary_margin
  score_survival

In that case, use scripts/rescore_fair_grid_live_profiles.py instead of
rerunning IsaacGym.

Do rerun IsaacGym if a reward candidate changes metric definitions, adds a new
primitive measurement that was not recorded, changes the scene/dynamics, changes
the action grid, or requires a different rollout policy.
```

Fair-audit script correction:

```text
scripts/evaluate_gait_target_fairness.py now has:
  --selection-score-key neutral_score
  --selection-score-key weighted_metric_reward_mean

For canonical reward validation, use:
  --selection-score-key weighted_metric_reward_mean

This ensures best_by_task_speed_gait.csv, best_by_task_speed.csv, summary.md,
and soft distributions select each gait's best continuous parameters under the
live canonical reward instead of the old neutral_score.
```

Recommended next fair-search command:

```text
Use batch_size=128 first. In this script the actual IsaacGym env count is:

effective_num_envs = batch_size * repeats_per_config

The previous batch_size=384 meant 768 envs and segfaulted during PhysX/env
initialization on flat_trot_efficiency vx=0.5.
```

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluate_gait_target_fairness.py \
  --training-range \
  --grid-mode action-space \
  --batch-size 128 \
  --repeats-per-config 2 \
  --steps 500 \
  --warmup-steps 100 \
  --reward-profile canonical_efficiency_candidate \
  --selection-score-key weighted_metric_reward_mean \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid \
  --skip-existing
```

After that fair search completes, compare multiple live reward profiles without
rerunning IsaacGym:

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw
python3 scripts/rescore_fair_grid_live_profiles.py \
  --input runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/fair_gait_grid_results.csv \
  --profiles canonical_efficiency_candidate,canonical_balanced_candidate \
  --output-dir runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/live_profile_rescore
```

Rejected command for now:

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

## 2026-06-17 Canonical Fair-Grid Result

The corrected canonical reward path has passed the same-trajectory
online/offline consistency check:

```text
runs/high_level_oracle_gait/reward_consistency/20260616_canonical_efficiency_candidate_broad

96 cases
39 metrics
max_abs_error = 0
passed = true
```

The corrected fair continuous-parameter search has also completed:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid

training_range = true
grid_mode = action-space
batch_size = 128
repeats_per_config = 2
steps = 500
warmup_steps = 100
reward_profile = canonical_efficiency_candidate
selection_score_key = weighted_metric_reward_mean
```

Important protocol point:

```text
This result ranks gait families only after every task-speed-gait receives the
same continuous-parameter search budget. It is therefore the fair comparison we
wanted, not the old fixed-residual/default-template comparison.
```

Canonical efficiency candidate result:

```text
top gait counts across 17 task-speed points:
  pronking = 12
  pacing = 3
  trotting = 2

flat:
  0.5 pronking > pacing
  1.0 pronking > pacing
  1.5 pronking > trotting
  2.0 pronking > trotting

ramp:
  0.5 pronking > pacing
  1.0 pronking > trotting
  1.5 pronking > trotting
  2.0 pronking > trotting

rough:
  0.5 pronking > pacing
  1.0 pronking > pacing
  1.5 pronking > trotting
  2.0 pacing > trotting

push:
  1.2 pronking > pacing
  1.5 trotting > pronking
  1.8 trotting > pronking

stones:
  1.7 pacing > pronking
  2.0 pacing > pronking

mean top1-top2 margin = 0.0161
```

Metric-gap readout:

```text
The efficiency candidate is not merely selecting pronking because continuous
parameters were fixed. Even after per-gait parameter search, pronking often wins
through combinations of progress, energy score, impact score, slip, and lateral
drift scores. At higher flat speeds it can lose progress to trotting but still
win the weighted objective through other terms.

Push and stones are weak-margin cases. Push 1.5/1.8 favors trotting only
narrowly. Stones favors pacing, also with small margins.
```

The same completed grid was then re-scored offline for
`canonical_balanced_candidate`:

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
Both canonical efficiency and canonical balanced are still pronking-dominant.
Balanced reduces the margins but does not solve the collapse. Neither candidate
is validated for PPO training.

This should be described as a pronking-dominant reward landscape, not yet as
PPO pronking collapse. PPO has not been trained on this candidate.

Engineering prior:
  canonical_efficiency_candidate is not a plausible final reward in its current
  form. WTW is known to be strong at trot, and trot is the expected efficient
  baseline in most ordinary locomotion settings. A broad pronking preference is
  therefore a warning sign. Held-out validation is used to diagnose why this
  happens, not to justify using this candidate for PPO.
```

Next action:

```text
Before revising reward weights, validate the top fair-grid parameter configs on
held-out seeds. The fair search currently selects the max-scoring config per
gait; even with equal search budget, this can overestimate a high-variance gait.

For every task-speed-gait, take the top-k parameter configs from the fair grid
(suggested k=5), then re-evaluate them with new random seeds and higher repeats.
Report:

- validated best mean;
- top-3 mean;
- standard deviation;
- fall rate;
- worst-tail or CVaR-style score;
- raw metric table and normalized score table.

Implementation rule:

```text
Held-out validation should be a configuration-level metrics cache:

  task + speed + gait + parameter config + seed -> raw metrics + score metrics

It should not be tied to a single reward name. A config that has already been
validated on held-out seeds can be re-scored offline by any future reward that
uses the saved raw/score metrics.
```

Efficient workflow:

```text
1. Use the completed fair grid to try many unified reward weight sets offline.
2. Keep only plausible candidates.
3. For each candidate, take top-k configs per task-speed-gait.
4. Deduplicate the union of selected configs.
5. Run held-out validation only for configs missing from the cache.
6. Re-score cached held-out metrics offline for all current and future candidates.
```

Implemented tools:

```text
scripts/select_heldout_gait_configs.py
  reads a completed fair grid;
  scores it with one or more reward profiles and/or existing score columns;
  selects top-k configs per task-speed-gait;
  deduplicates by task, speed, gait, and residual parameters;
  writes heldout_config_requests.csv and new_heldout_config_requests.csv.

scripts/evaluate_gait_target_fairness.py --config-csv ... --eval-from-config
  runs only the requested held-out configs;
  records validation_seed in every output row;
  produces the same metric columns as the original fair grid.
```

Current held-out request set:

```text
runs/high_level_oracle_gait/heldout_config_selection/20260617_topk_union_k3
runs/high_level_oracle_gait/heldout_config_selection/20260617_topk_union_k5

input:
  fair_target_gait_audit/20260617_canonical_efficiency_action_grid/fair_gait_grid_results.csv

score keys:
  canonical_efficiency_candidate_score
  canonical_balanced_candidate_score
  neutral_score
  weighted_metric_reward_mean

top_k_per_task_speed_gait = 3
selected_unique_configs = 405
by gait:
  pronking = 108
  trotting = 105
  bounding = 97
  pacing = 95

top_k_per_task_speed_gait = 5
selected_unique_configs = 658
by gait:
  pronking = 173
  trotting = 171
  bounding = 157
  pacing = 157
```

First held-out validation command:

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

Held-out validation result:

```text
runs/high_level_oracle_gait/heldout_validation/20260617_topk_union_k3_seed101

request set:
  top-k union k=3
  405 configs
  17 task-speed points
  validation_seed = 101
  repeats_per_config = 4
```

Canonical efficiency live reward ranking:

```text
original fair-grid best:
  pronking = 12
  pacing = 3
  trotting = 2

held-out seed101 best:
  pronking = 14
  pacing = 2
  trotting = 1

held-out top-3-config mean:
  pronking = 14
  pacing = 2
  trotting = 1
```

Interpretation:

```text
The pronking-dominant reward landscape survives held-out validation and is not
only a single best-config max-over-grid artifact. This strengthens the decision
that canonical_efficiency_candidate is not suitable for PPO.

This is enough to stop further large-scale validation of
canonical_efficiency_candidate. The next issue is not weight tuning around this
candidate, but redesigning the universal metric definitions and compensation
structure.
```

Neutral score contrast:

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

This contrast means pronking dominance is specific to the canonical live reward
candidate/weights/metric definitions, not an unavoidable property of the
candidate trajectories.

Important caveat from this run:

```text
The ranking by weighted_metric_reward_mean is valid, because it comes directly
from HighLevelGaitWrapper reward terms. However, some diagnostic raw columns in
evaluate_gait_target_fairness.py were computed from post-step env buffers rather
than the wrapper's reward primitives. In particular, energy diagnostics such as
transport_cost_proxy and the old torque_penalty_mean can disagree with
score_energy.

The evaluator has been patched to log canonical raw primitives from wrapper
reward terms where available. Re-run a small diagnostic held-out subset before
making detailed energy/impact/slip/scuff claims.
```

Next metric-design plan:

```text
Do not try to force old target labels or manually make trot win everywhere.
Trot should be treated as a strong plausibility prior in ordinary flat/ramp/rough
conditions because WTW is known to be strong at trot and trot is the expected
efficient baseline in many standard locomotion settings. But push, stepping
stones, and speed-dependent edge cases should remain data-driven.

The objective should be redesigned as a universal physical reward with less
linear compensation:

1. survival and command tracking are primary requirements;
2. orientation, slip, impact, and scuffing/contact safety are safety constraints;
3. energy and smoothness optimize after locomotion quality is acceptable.
```

Candidate structure:

```text
quality_gate = survival_score * tracking_score

reward =
  tracking_weight * tracking_score
  + survival_weight * survival_score
  + quality_gate * (
      stability_terms
      + contact_safety_terms
      + efficiency_terms
      + smoothness_terms
    )
```

Metric items to fix before another full fair grid:

```text
energy:
  keep mean mechanical power as a diagnostic;
  add or replace the training energy score with a stable cost-of-transport-style
  proxy using a protected denominator, e.g. max(forward_distance, epsilon);
  avoid rewarding low movement or poor tracking as "efficient".

impact:
  add landing-event statistics such as impact velocity/force peak, impulse, or
  high percentiles. Whole-rollout averages can hide short high-impact contacts.

slip:
  normalize by meaningful contact time/contact force and inspect slip per
  distance. Short-contact gaits should not get a free low-slip score.

progress/tracking:
  make failure to follow commanded speed difficult to compensate with energy,
  slip, or contact terms.
```

Simulation reuse rule for the next phase:

```text
If only weights change, use existing fair/held-out CSVs.

If score normalization changes and required raw primitives are already saved,
re-score offline.

If energy/impact/slip/scuff definitions require primitives not saved in the CSV,
run a small representative metric sanity audit first. Rerun the full fair grid
only after the new metric definitions are stable and visually/numerically sane.
```

Rerun rule:

```text
Changing only reward weights:
  no full fair-grid rerun;
  no held-out rerun for already cached configs.

Changing score normalization:
  offline re-score is enough only if the needed raw metric is saved.

Changing metric definitions, terrain, push protocol, rollout length, action
grid, gait template mapping, or WTW low-level checkpoint:
  rerun the affected simulation data because the trajectory or required
  primitive measurements have changed.
```

Only after this held-out validation should we decide whether pronking dominance
is a real physical result or a metric/normalization/max-selection artifact.

If pronking remains genuinely best on tracking, stability, energy, impact,
scuffing/contact safety, slip, and survival, a pronk-heavy policy may be a
valid unified-reward outcome. If pronking wins through score artifacts or
unphysical loopholes, revise universal metric definitions/scales/weights.

After revising a candidate, re-score the existing grid if the metric definitions
are unchanged. Rerun IsaacGym fair search only if metric definitions, scenes,
action grid, or rollout protocol change.
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

## 2026-06-17 Current Metric Sanity Protocol

This section supersedes the older fixed-gait reward-v5 branch above. Do not
start PPO and do not run another full fair grid until the metric definitions
pass a small sanity audit.

Current decision:

```text
Stop canonical_efficiency_candidate as a PPO reward candidate.
Do not continue large-scale validation of that candidate.
Repair metric definitions and compensation structure first.
```

Why:

```text
canonical_efficiency_candidate:
  held-out seed101 best config ranking -> pronking 14 / 17

neutral_score on the same held-out trajectories:
  best config ranking -> trotting 10 / 17
```

This means the trajectory library itself is not inevitably pronk-dominant. The
problem is the live reward metric definition and weighting, not simply unfair
continuous-parameter search.

### What to check next

The small sanity audit should answer whether the reward scores and raw physical
metrics move in sensible directions:

```text
tracking:
  Does the score-best gait actually track commanded velocity?

energy:
  Does low torque_penalty / transport_cost_proxy correspond to high energy score?
  Are we accidentally rewarding "moving less" as "efficient"?

impact:
  Does impact reflect landing/contact events rather than a diluted rollout mean?

slip:
  Does slip avoid giving short-contact gaits an unfair advantage?

scuffing/contact safety:
  Is the current scuffing proxy meaningful enough to trust?
```

### New tools

```text
scripts/select_metric_sanity_configs.py
```

Selects a compact representative config CSV from an existing fair grid. This is
only a sampling tool; it does not produce final gait rankings.

```text
scripts/analyze_metric_sanity_audit.py
```

Analyzes a small sanity simulation and writes:

```text
best_by_task_speed_gait.csv
score_best_by_task_speed.csv
raw_score_direction_checks.csv
score_best_compensation_flags.csv
summary.md
```

### Representative config selection already created

```text
runs/high_level_oracle_gait/metric_sanity/20260617_config_selection
```

Source:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260617_canonical_efficiency_action_grid/fair_gait_grid_results.csv
```

Selection:

```text
task-speed points:
  flat_trot_efficiency:1.0
  ramp_up_trot_robustness:1.0
  rough_slope_trot_robustness:1.0
  push_lateral_pace_recovery:1.5
  stepping_stones_easy_bound_highspeed:2.0

score keys:
  weighted_metric_reward_mean
  neutral_score

top_k:
  1 per task-speed-gait per score key

selected_unique_configs:
  37
```

Gait balance:

```text
bounding: 9
pacing: 10
pronking: 9
trotting: 9
```

### Small simulation command

```bash
cd /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
conda activate go2_wtw

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

Analyze it:

```bash
python3 scripts/analyze_metric_sanity_audit.py \
  --input runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/fair_gait_grid_results.csv \
  --score-key weighted_metric_reward_mean \
  --output-dir runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/analysis
```

### Small sanity result

Run:

```text
runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202
```

Config:

```text
37 requested configs
5 representative task-speed points
repeats_per_config = 4
steps = 500
warmup_steps = 100
seed = 202
reward_profile = canonical_efficiency_candidate
selection_score_key = weighted_metric_reward_mean
```

Live weighted reward best gait:

```text
flat_trot_efficiency vx=1.0: pronking
ramp_up_trot_robustness vx=1.0: pronking
rough_slope_trot_robustness vx=1.0: pronking
push_lateral_pace_recovery vx=1.5: trotting
stepping_stones_easy_bound_highspeed vx=2.0: pacing
```

Representative raw metric comparison:

```text
flat 1.0:
  live best pronking, but trotting has lower vx error and lower torque penalty.

ramp 1.0:
  live best pronking, but pacing has lower torque penalty and trotting is close.

rough 1.0:
  live best pronking, but trotting has lower vx error and lower torque penalty.

push 1.5:
  live best trotting, also best on tracking, torque, and slip among this small
  config set.

stones 2.0:
  live best pacing, but all gaits have large vx error; this point is not a clean
  success case for any gait.
```

Analysis output:

```text
runs/high_level_oracle_gait/metric_sanity/20260617_small_sanity_seed202/analysis
```

It reports:

```text
raw/score direction disagreements = 9
score-best gait counts:
  pronking 3
  trotting 1
  pacing 1
```

Important interpretation:

```text
The direction disagreements do not by themselves prove a code bug. The current
CSV compares averaged raw penalties against averaged nonlinear per-step scores.
Because E[exp(-x)] is not the same object as exp(-E[x]), gait ranking can change
when penalty variance differs. However, this confirms that current score terms
cannot be casually interpreted as "lower average energy/slip/impact".

canonical_efficiency_candidate remains rejected for PPO. The next step is to
repair metric definitions and aggregation, especially tracking gating, energy
definition, slip/contact normalization, and impact event statistics.
```

### Reuse rule

If only reward weights change, reuse existing fair-grid and held-out CSVs by
offline re-scoring.

If only score normalization changes and the required raw primitive is already
saved, reuse existing CSVs by offline re-scoring.

If the metric definition needs primitives not present in the CSV, or changes
the simulation protocol, terrain, action grid, gait template, or low-level WTW
checkpoint, run a small sanity simulation first. Only rerun the full fair grid
after the metric definition passes this sanity layer.

## 2026-06-18 Metric Repair Direction

The current bottleneck is metric definition and temporal aggregation, not reward
weight tuning.

The GPT critique is accepted with the following bounded interpretation:

```text
Correct:
  The 37-config sanity audit is enough to reject canonical_efficiency_candidate
  for PPO.

Correct:
  E[exp(-x)] and exp(-E[x]) are different; averaged nonlinear per-step scores
  can rank gaits differently from averaged raw penalties.

Correct:
  Tracking should be a primary task requirement and should not be fully
  compensated by energy, slip, impact, or scuffing.

Correction:
  This does not prove that trotting is always the final optimum. It only shows
  that the current unified reward does not integrate tracking, energy, slip,
  impact, and safety in a physically convincing way.
```

Next implementation principles:

```text
tracking / survival:
  make them base terms. A gait that does not follow the command should not win
  mainly by looking cheap or safe.

quality gate:
  use tracking quality to gate secondary terms, but keep a floor so early PPO
  training does not become reward-sparse.

energy:
  record and inspect at least:
    - torque penalty
    - mean absolute mechanical power sum(|tau * qdot|)
    - stabilized transport-cost proxy
  Do not rely on a single energy score until its physical meaning is clear.

slip:
  normalize by meaningful contact time/contact force. Record contact-time slip
  and consider per-distance slip so short-contact gaits do not win by avoiding
  contact samples.

impact:
  move toward landing-event statistics or high-percentile contact/impact
  quantities. Whole-rollout averages can hide short, high-impact landings.

scuff/contact safety:
  keep current scuff proxy as diagnostic only until the terrain-relative
  primitive is proven meaningful.
```

Validation order:

```text
1. modify canonical metric primitives and score aggregation;
2. run same-trajectory online/offline consistency;
3. rerun the same 37-config sanity audit;
4. inspect raw metric, score, and contribution tables;
5. only if sanity passes, run a small grid;
6. then full fair grid;
7. then held-out top-k validation;
8. only then PPO.
```

Do not use "which gait won" as the only acceptance criterion. A candidate is
acceptable only if the winning gait can be explained by raw tracking, stability,
contact safety, energy, and survival metrics without relying on an obvious
aggregation loophole.

### Implemented v2 diagnostic candidate

The code now contains:

```text
canonical_efficiency_v2_candidate
```

Status:

```text
diagnostic_only_unvalidated_candidate
```

It must not be used for PPO until it passes the validation sequence below.

Design:

```text
Base task terms:
  progress
  yaw_tracking
  survival

Tracking-quality gate:
  tracking_gate = clamp(0.25 + 0.75 * progress, 0.25, 1.0)

Gated secondary terms:
  gated_orientation
  gated_lateral_drift
  gated_contact_slip
  gated_impact
  gated_scuffing
  gated_power_efficiency
  gated_transport_efficiency
  gated_action_smoothness
```

New raw primitives:

```text
contact_slip_penalty:
  sum(contact * foot_xy_speed_squared) / contact_count

mechanical_power_abs:
  sum(abs(torque * joint_velocity))

transport_cost_proxy:
  mechanical_power_abs / max(abs(base_vx), 0.3)
```

These are intentionally still proxies. Impact is still step-level landing
velocity RMS rather than a full P90/P95 event statistic. Therefore v2 is a
metric-repair candidate, not a final reward.

Required next validation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/check_high_level_reward_consistency.py \
  --reward-profile canonical_efficiency_v2_candidate \
  --output-dir runs/high_level_oracle_gait/reward_consistency/20260618_canonical_efficiency_v2_candidate
```

If consistency passes:

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

### v2 consistency result

Run:

```text
runs/high_level_oracle_gait/reward_consistency/20260618_canonical_efficiency_v2_candidate
```

Result:

```text
reward_profile = canonical_efficiency_v2_candidate
tasks = flat_trot_efficiency, stepping_stones_easy_bound_highspeed
gaits = pronking, trotting, bounding, pacing
residual_sets = zero, high_clearance
cases = 16
metrics = 54
max_abs_error = 0
passed = True
```

Interpretation:

```text
The online `HighLevelGaitWrapper` reward terms and offline recomputation from
the same recorded primitives are exactly aligned for v2. This validates the
formula path only. It does not validate whether v2 is physically reasonable or
whether its gait ranking is acceptable.
```

### v2 small metric sanity result

Run:

```text
runs/high_level_oracle_gait/metric_sanity/20260618_v2_small_sanity_seed203
```

Protocol:

```text
reward_profile = canonical_efficiency_v2_candidate
seed = 203
configs = same 37 representative sanity configs selected from the previous
          fair grid
ranking_score_key = weighted_metric_reward_mean
```

Score-best ranking:

```text
flat_trot_efficiency vx=1.0 -> trotting
push_lateral_pace_recovery vx=1.5 -> trotting
ramp_up_trot_robustness vx=1.0 -> pronking
rough_slope_trot_robustness vx=1.0 -> trotting
stepping_stones_easy_bound_highspeed vx=2.0 -> pacing
```

Score-best counts:

```text
trotting = 3 / 5
pronking = 1 / 5
pacing = 1 / 5
bounding = 0 / 5
```

Interpretation:

```text
v2 removes the main failure mode of canonical_efficiency_candidate on this
small regression set: there is no longer broad pronking dominance. This means
the tracking-gated contact/efficiency repair is directionally useful.

However, v2 is still not validated for full fair-grid reruns or PPO:

1. flat is effectively a trot/pronk tie;
2. rough is also effectively a tie, but the tiny trotting advantage is largely
   caused by action_boundary_margin rather than core locomotion physics;
3. ramp still prefers pronking because of contact_slip, progress, yaw, and
   impact terms despite worse power/transport than trotting/pacing;
4. stones prefers pacing under the current metrics, but all gaits still have
   poor high-speed tracking there.
```

Mechanism-level interpretation:

```text
The old canonical_efficiency_candidate used a mostly linear weighted average:
progress, orientation, slip, energy, impact, scuffing, smoothness, boundary,
and survival could compensate for each other directly. On the small sanity set,
this produced pronking wins on flat/ramp/rough.

v2 changes the reward structure rather than merely retuning weights:

1. progress weight is increased and treated as a primary task term;
2. survival remains a primary base term;
3. secondary terms are gated by tracking_gate, so poor velocity tracking reduces
   the value of orientation/contact/efficiency/smoothness scores;
4. contact-normalized slip, absolute mechanical power, and a stabilized
   transport-cost proxy are logged and scored separately;
5. energy is no longer represented only by the old torque score.
```

Observed mechanism effects on the same 37-config sanity set:

```text
old canonical candidate:
  pronking = 3 / 5
  trotting = 1 / 5
  pacing = 1 / 5

v2 candidate:
  trotting = 3 / 5
  pronking = 1 / 5
  pacing = 1 / 5
```

Representative contribution changes:

```text
flat vx=1.0:
  old top = pronking.
  v2 top = trotting, but only by a near-zero margin.
  Trotting gains from progress/tracking and gated power/transport efficiency;
  pronking still gains from yaw, lateral/contact slip, and impact.

rough vx=1.0:
  old top = pronking.
  v2 top = trotting, but mostly because pronking has poor action_boundary_margin.
  This is not a valid physical explanation and motivates removing or weakening
  action_boundary_margin from the reward/ranking term.

ramp vx=1.0:
  old top = pronking.
  v2 top remains pronking.
  Pronking gains from contact_slip, progress, yaw, and impact, while
  trotting/pacing are better on power/transport efficiency. This case still
  needs metric scrutiny.
```

Decision:

```text
Do not promote canonical_efficiency_v2_candidate to PPO.
Do not run a full fair grid yet.
Do not make v3 a one-line boundary-margin tweak. The next candidate must split
physical locomotion reward from action regularization.

Fair gait audit should rank gait/configs using only R_physical:
  tracking/progress, survival, orientation, lateral control, contact-normalized
  slip, impact, scuffing/contact safety, and one primary energy-efficiency term.

Action-health terms should remain logged but should not decide fair gait
ranking:
  action_boundary_margin, action_magnitude, action_smoothness, and possibly
  gait_stability.

PPO may later use:
  R_total = R_physical + lambda_reg * R_regularization
with lambda_reg small enough that regularization cannot dominate physical task
success.

Then rerun same-trajectory consistency and the same 37-config sanity audit.
```

Acceptance condition for the next step:

```text
Do not require a specific gait to win every point.
Require that score-best gaits can be explained by tracking, survival,
contact-normalized slip, power/transport efficiency, impact, scuffing, and
lateral/orientation metrics without obvious contradiction.
```

Additional v3 design constraints:

```text
1. Do not double-count efficiency too aggressively. Keep mechanical power as
   the main reward energy term first; keep transport_cost_proxy as a diagnostic
   until its low-speed/push behavior is stable.

2. When reporting gated terms, always report:
   raw primitive, normalized score, tracking_gate, and weighted contribution.
   A high gated contribution may come from better tracking rather than a better
   raw energy/slip/impact primitive.

3. For stones and other hard points, do not call a relative winner "good" if
   all gaits fail the basic tracking/survival threshold. Mark those as
   task-quality failures first.
```

### Implemented v3 physical diagnostic profile

The code now contains:

```text
canonical_efficiency_v3_physical
```

Status:

```text
diagnostic_only_unvalidated_candidate
```

Purpose:

```text
Use this profile for fair gait audit physical ranking. It is not a PPO reward.
It intentionally excludes action regularizers from the score that ranks gait
families.
```

Included physical terms:

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

Excluded from physical ranking:

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
action_boundary_margin/action_magnitude/action_smoothness/gait_stability are
policy action-health or regularization diagnostics. They can be logged and may
later be added to PPO with a small lambda_reg, but they should not decide which
gait is physically better in fair audits.

transport_cost_proxy remains logged but is not a v3 reward term yet, because
low-speed, push recovery, and short-window behavior can make distance-normalized
energy unstable. Use mechanical power first as the main energy term.
```

Analysis update:

```text
scripts/analyze_metric_sanity_audit.py
```

now accepts:

```text
--reward-profile canonical_efficiency_v3_physical
```

and writes:

```text
weighted_contribution_decomposition.csv
top_vs_second_contribution_gaps.csv
```

These contribution files record:

```text
raw primitive
normalized score
tracking_gate
metric weight
weighted contribution
top-vs-second weighted gap
```

This is required for interpreting gated scores: a high gated contribution may
come from better tracking_gate, not a better raw energy/slip/impact primitive.

### v3 consistency result

Run:

```text
runs/high_level_oracle_gait/reward_consistency/20260619_canonical_efficiency_v3_physical
```

Result:

```text
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
The online `HighLevelGaitWrapper` v3 physical reward terms and offline
recomputation from the same recorded primitives are exactly aligned for the
checked cases.

This only validates formula consistency. It does not validate that v3 is a good
reward, nor does it validate gait ranking.
```

Next required validation:

```text
Rerun the same 37-config metric sanity audit with
canonical_efficiency_v3_physical, then analyze with contribution decomposition.
Do not run a full fair grid or PPO before this sanity layer is reviewed.
```

### v3 physical small metric sanity result

Run:

```text
runs/high_level_oracle_gait/metric_sanity/20260619_v3_physical_small_sanity_seed204
```

Protocol:

```text
reward_profile = canonical_efficiency_v3_physical
seed = 204
configs = same 37 representative sanity configs
ranking_score_key = weighted_metric_reward_mean
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

Score-best counts:

```text
pronking = 3 / 5
trotting = 1 / 5
pacing = 1 / 5
bounding = 0 / 5
```

Main readout:

```text
v3 succeeds at one narrow goal: action_boundary_margin and other action
regularizers no longer decide the fair-gait ranking.

v3 fails the broader sanity check: once the action regularizers are removed,
the remaining physical score still prefers pronking on flat/ramp/rough.
```

Contribution evidence:

```text
flat vx=1.0:
  pronking beats trotting by 0.0072.
  Trotting is better on progress/tracking and mechanical power.
  Pronking wins through lateral drift, contact slip, yaw, orientation, scuffing,
  and impact.

ramp vx=1.0:
  pronking beats trotting by 0.0235.
  Pronking wins on progress, contact slip, orientation, yaw, impact, and
  scuffing; trotting is better on mechanical power.

rough vx=1.0:
  pronking beats trotting by 0.0067.
  Pronking wins mainly through contact slip and impact, while trotting is
  slightly better on progress/yaw/lateral/scuffing.

push vx=1.5:
  trotting beats pronking by 0.0320, mainly through progress/tracking and
  orientation. This is the cleanest point.

stones vx=2.0:
  pacing barely beats trotting by 0.0050, mostly through progress. All gaits
  still have large tracking error, so this is a task-quality warning rather
  than a strong gait conclusion.
```

Decision:

```text
Do not promote canonical_efficiency_v3_physical.
Do not run full fair grid or PPO with v3.

The next problem is physical-term compensation: secondary physical terms can
still compensate for worse tracking and power. The next reward candidate should
use a tracking-first or constraint-style structure, not just another linear
weighted average.
```

Candidate direction for v4:

```text
1. Keep action regularizers outside physical fair ranking.
2. Make tracking/progress a primary requirement:
   - either a stronger progress term,
   - or a minimum tracking/progress gate before secondary terms contribute,
   - or a two-stage/constraint-style score.
3. Prevent lateral/contact/impact/scuffing terms from collectively overriding
   worse command tracking on ordinary flat/ramp/rough.
4. Keep mechanical power as the main energy reward term.
5. Keep transport_cost_proxy diagnostic-only until stable.
6. Mark high-error stones cases as task-quality failures before treating the
   relative winner as meaningful.
```

### 2026-06-20 v4 design decision

The GPT critique is accepted with a tighter interpretation:

```text
v3 ruled out action-regularizer contamination, but showed that the physical
reward still permits secondary terms to compensate for weaker tracking and
power. Therefore v4 should change reward structure, not merely retune linear
weights.
```

v4 should use a tracking-first / constraint-style physical score:

```text
R_physical_v4 =
  base_task_terms
  + safety_constraint_terms
  + tracking_gate_strict * efficiency_terms
```

Base task terms:

```text
progress / velocity tracking:
  primary term. This is the commanded locomotion objective and should dominate
  ordinary flat/ramp/rough comparisons.

survival:
  always active.

orientation:
  always active but not strong enough to override clear tracking and power
  advantages by itself.

yaw_tracking:
  retained with low weight. In v3 it helped pronking on flat; it should not
  dominate forward locomotion quality.
```

Safety constraint terms:

```text
contact_slip
impact
scuffing
lateral_drift
```

These should not be unrestricted linear bonuses. They should be thresholded or
saturated:

```text
safety_constraint(score; low, high) = clamp((score - low) / (high - low), 0, 1)
```

Reason:

```text
If a gait is unsafe, it should be penalized. But once several gaits are already
within an acceptable safety band, tiny differences in slip/impact/scuff/lateral
should not collectively overpower better tracking and lower power.
```

Efficiency terms:

```text
mechanical_power / power_efficiency:
  the main energy term for v4.

transport_cost_proxy:
  diagnostic-only until low-speed, push, and short-window behavior is stable.
```

Efficiency should be gated by stricter tracking quality:

```text
tracking_gate_strict = clamp((progress - progress_threshold) /
                             (1 - progress_threshold), 0, 1)
```

Suggested initial threshold:

```text
progress_threshold = 0.70 to 0.75
```

Reason:

```text
Poorly tracked motion should not win by being cheap. Once command tracking is
reasonable, power efficiency can decide between otherwise acceptable gaits.
```

Initial v4 weight intent:

```text
progress: strong
survival: strong but usually saturated
orientation: moderate
yaw_tracking: weak
safety constraints: moderate and saturated
gated_power_efficiency: moderate, active mainly after tracking is acceptable
action regularizers: excluded from fair ranking
transport_efficiency: excluded from reward, logged only
```

Do not use v4 to force trot:

```text
The acceptance criterion is not "trot wins N/5". The acceptance criterion is:
the winning gait must be explainable from primary task quality and real physical
metrics, and small secondary advantages must not override clearly better
tracking and power in ordinary locomotion scenes.
```

Expected sanity behavior:

```text
flat/ramp/rough:
  trot should be a strong baseline. If pronking wins, it must win by a clear
  primary-task or safety margin, not by many small secondary bonuses.

push:
  data decides. Trot, pace, or pronk can be acceptable if tracking/recovery,
  orientation, safety, and power support it.

stones:
  if every gait has large vx_err, mark the point as task-quality failure before
  interpreting the relative winner as a meaningful gait preference.
```

Next implementation sequence:

```text
1. Add threshold/saturation helper scores to high_level_reward_metrics.py.
2. Add canonical_efficiency_v4_physical as diagnostic-only.
3. Keep action regularizers excluded from v4 physical ranking.
4. Keep transport_cost_proxy diagnostic-only.
5. Run same-trajectory online/offline consistency.
6. Run the same 37-config sanity audit.
7. Analyze with contribution decomposition.
8. Only if sanity passes, proceed to a small fair grid.
```

### 2026-06-20 v4 implementation update

`canonical_efficiency_v4_physical` has been implemented in
`go2_gym/envs/wrappers/high_level_reward_metrics.py` and registered as
diagnostic-only in `scripts/train_high_level_oracle_ppo.py`.

The profile is:

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

New canonical score terms:

```text
tracking_gate_strict = clamp((progress - 0.75) / 0.25, 0, 1)

safety_lateral_drift = clamp((lateral_drift_score - 0.25) / 0.25, 0, 1)
safety_contact_slip = clamp((contact_slip_score - 0.60) / 0.25, 0, 1)
safety_impact = clamp((impact_score - 0.80) / 0.12, 0, 1)
safety_scuffing = clamp((scuffing_score - 0.70) / 0.20, 0, 1)

strict_gated_power_efficiency =
  tracking_gate_strict * power_efficiency_score
```

Design reason:

```text
1. progress/tracking is now structurally dominant, not merely one linear term.
2. power efficiency cannot help a gait until tracking is reasonably good.
3. contact slip, impact, scuffing, and lateral drift are safety constraints:
   bad values hurt, but tiny differences inside the acceptable band saturate.
4. action regularizers do not participate in fair gait ranking.
5. transport_cost_proxy remains a logged diagnostic, not a reward term.
```

This implementation is not validated for PPO. It is only ready for:

```text
same-trajectory online/offline consistency
37-config small metric sanity
contribution decomposition
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

Review criteria after the sanity audit:

```text
1. tracking-bad configs should not win by power/contact/safety compensation;
2. winner must be explainable from raw primitives and weighted contributions;
3. thresholded safety terms must not become a new hidden dominant factor;
4. flat/ramp/rough should make trot a strong baseline unless another gait has
   a clear primary-task or safety advantage;
5. stones with large vx_err should be marked as task-quality weak before using
   its relative winner as a target.
```

### 2026-06-20 v4 consistency result

Run:

```text
runs/high_level_oracle_gait/reward_consistency/20260620_canonical_efficiency_v4_physical
```

Result:

```text
reward_profile = canonical_efficiency_v4_physical
tolerance = 1e-05
max_abs_error = 0
passed = True
metric comparison rows = 960
```

Interpretation:

```text
The v4 online `HighLevelGaitWrapper` reward terms and offline recomputation from
the same recorded trajectory primitives agree exactly. This clears the formula
consistency gate.

It does not validate the reward design itself. v4 is still diagnostic-only until
the same 37-config metric sanity audit and contribution decomposition show that
the ranking is physically explainable.
```

Immediate next command:

```bash
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
```

### 2026-06-20 v4 small metric sanity result

Run:

```text
runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205
```

Analysis:

```text
runs/high_level_oracle_gait/metric_sanity/20260620_v4_physical_small_sanity_seed205/analysis
```

Summary:

```text
rows in best-by-task-speed-gait analysis = 20
task-speed points = 5
score_best_warning_count = 0

winner counts:
  trotting: 2
  pronking: 2
  pacing: 1
```

Winners:

```text
flat_trot_efficiency vx=1.0:
  trotting
  score = 0.8751
  vx_err = 0.1256
  power = 143.1

push_lateral_pace_recovery vx=1.5:
  trotting
  score = 0.7540
  vx_err = 0.3138
  power = 214.8

ramp_up_trot_robustness vx=1.0:
  pronking
  score = 0.8578
  vx_err = 0.1455
  power = 167.3

rough_slope_trot_robustness vx=1.0:
  pronking
  score = 0.8239
  vx_err = 0.1906
  power = 147.4

stepping_stones_easy_bound_highspeed vx=2.0:
  pacing
  score = 0.5780
  vx_err = 0.6033
  power = 315.2
```

Contribution interpretation:

```text
flat:
  trotting beats pronking by 0.0085.
  Main positive terms are progress, orientation, and strict gated power.
  Pronking still has better yaw/impact/scuffing, but those no longer override
  trotting's tracking and power. This is the intended v4 behavior.

push:
  trotting beats bounding by 0.0319.
  Main positive terms are progress, orientation, contact-slip safety, and
  strict gated power. This is physically interpretable.

ramp:
  pronking beats trotting by 0.0118.
  Tracking is almost tied, while pronking has better yaw, survival, lateral,
  contact slip, impact, and scuffing. Trotting has better orientation and
  power. This is acceptable as a candidate result, not an obvious failure.

rough:
  pronking beats trotting by 0.0100.
  This remains the least clean ordinary-terrain result. Pronking wins through
  orientation, impact, lateral, contact-safety score, and gated power, while
  trotting or pacing are competitive on tracking/contact raw metrics. Recheck
  rough in the next audit before accepting the reward.

stones:
  pacing beats trotting by 0.0341.
  The main winner driver is progress, but all gaits have high vx_err. This
  point should be treated as task-quality weak before interpreting the relative
  winner as a gait target.
```

Decision:

```text
v4 passes the same-trajectory consistency gate and the first 37-config small
sanity gate. It is better than v3 and no longer shows the obvious action
regularizer or broad linear-compensation failure.

v4 is still not validated for PPO. Do not run PPO yet.
The next step is a small fair-grid audit under v4, with special attention to
rough_slope and stepping_stones.
```

### 2026-06-21 v4 representative action-grid result

Run:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260620_v4_physical_representative_action_grid
```

Protocol:

```text
eval points:
  flat_trot_efficiency:1.0
  ramp_up_trot_robustness:1.0
  rough_slope_trot_robustness:1.0
  push_lateral_pace_recovery:1.5
  stepping_stones_easy_bound_highspeed:2.0

grid_mode = action-space
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
batch_size = 64
repeats_per_config = 2
steps = 500
warmup_steps = 100
seed = 206
```

Best gait per task-speed:

```text
flat vx=1.0:
  trotting
  score = 0.8816
  vx_err = 0.1119
  power = 135.4

push vx=1.5:
  trotting
  score = 0.7721
  vx_err = 0.2845
  power = 193.8

ramp vx=1.0:
  pronking
  score = 0.8716
  vx_err = 0.1171
  power = 160.5

rough vx=1.0:
  pronking
  score = 0.8382
  vx_err = 0.1608
  power = 170.0

stones vx=2.0:
  pacing
  score = 0.5956
  vx_err = 0.6031
  power = 319.3
```

Gait ranking details:

```text
flat:
  trot 0.8816, pronk 0.8785, bound 0.8445, pace 0.8413.
  Trot wins by a small margin because tracking/power/lateral are better.
  Pronk still has better yaw/impact/scuff, but those no longer dominate.

push:
  trot 0.7721, pronk 0.7458, bound 0.7406, pace 0.7317.
  Trot is clearly preferred and has the best tracking, power, contact slip,
  and reasonable orientation. This is a clean result.

ramp:
  pronk 0.8716, trot 0.8543, pace 0.8244, bound 0.8190.
  Pronk has better tracking, yaw, lateral, impact, scuffing, and fall rate.
  Trot has better power. This is physically explainable and not an obvious
  reward failure.

rough:
  pronk 0.8382, trot 0.8357, pace 0.8073, bound 0.7957.
  This is effectively close. Trot has better neutral score, power, orientation,
  fall rate, and lateral; pronk has slightly better v4 score from progress,
  contact slip, impact, and strict gated power. Treat rough as uncertain, not
  as a strong pronk target.

stones:
  pace 0.5956, trot 0.5810, bound 0.5686, pronk 0.5619.
  Pace wins mainly through progress, but all gaits have high vx_err. This point
  remains task-quality weak and should not be converted into a strong gait
  prior or hard target.
```

Decision:

```text
The representative action-grid supports continuing v4 evaluation.
Do not train PPO yet.
Next run a v4 training-range fair audit to check the actual sampled-speed
coverage. If that remains physically interpretable, then run held-out top-k
validation for the selected configs before any PPO.
```

### 2026-06-21 v4 training-range action-grid result

Run:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid
```

Analysis:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid/analysis
```

Protocol:

```text
training-range speed points = 17
grid_mode = action-space
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
batch_size = 64
repeats_per_config = 2
steps = 500
warmup_steps = 100
seed = 207
```

Winner counts:

```text
trotting: 10
pronking: 6
pacing: 1
bounding: 0
```

Best gait per task-speed:

```text
flat_trot_efficiency:
  vx=0.5 -> trotting, score 0.9281, vx_err 0.0604
  vx=1.0 -> trotting, score 0.8807, vx_err 0.1043
  vx=1.5 -> trotting, score 0.8189, vx_err 0.1985
  vx=2.0 -> trotting, score 0.7529, vx_err 0.3412

push_lateral_pace_recovery:
  vx=1.2 -> trotting, score 0.7997, vx_err 0.2354
  vx=1.5 -> trotting, score 0.7678, vx_err 0.2883
  vx=1.8 -> trotting, score 0.7571, vx_err 0.3289

ramp_up_trot_robustness:
  vx=0.5 -> pronking, score 0.9167, vx_err 0.0705
  vx=1.0 -> pronking, score 0.8744, vx_err 0.1152
  vx=1.5 -> pronking, score 0.8076, vx_err 0.2360
  vx=2.0 -> trotting, score 0.7208, vx_err 0.4184

rough_slope_trot_robustness:
  vx=0.5 -> pronking, score 0.9001, vx_err 0.0816
  vx=1.0 -> pronking, score 0.8297, vx_err 0.1726
  vx=1.5 -> pronking, score 0.7617, vx_err 0.2901
  vx=2.0 -> trotting, score 0.6284, vx_err 0.5272

stepping_stones_easy_bound_highspeed:
  vx=1.7 -> trotting, score 0.6482, vx_err 0.4985
  vx=2.0 -> pacing, score 0.6080, vx_err 0.5312
```

Contribution interpretation:

```text
flat:
  Trotting wins all sampled speeds. Its advantage grows with speed and is mainly
  from progress/tracking and strict gated power. Pronk still often has better
  yaw/contact/impact, but these no longer overcompensate the primary task.

push:
  Trotting wins all sampled speeds. The wins are mostly explained by progress,
  orientation, and strict gated power. This strongly argues against using the
  old fixed push -> pace target under the current unified objective.

ramp:
  Pronking wins 0.5/1.0/1.5 and trotting wins 2.0. At 1.0-1.5 the pronking wins
  are physically interpretable because they are supported by progress/contact/
  impact terms, while trotting tends to retain better power. The low-speed 0.5
  margin is smaller and should be rechecked in held-out validation.

rough:
  Pronking wins 0.5/1.0/1.5 and trotting wins 2.0. The margins are modest,
  especially 0.5 and 1.0, and some wins are driven by safety terms rather than
  a clean primary-task advantage. Treat rough as uncertain until held-out top-k
  validation.

stones:
  Trotting wins 1.7 and pacing wins 2.0, but both have high vx_err. This task
  remains quality-limited and should not be turned into a strong gait target.
```

Decision:

```text
v4 now passes consistency, 37-config sanity, representative action-grid, and
training-range action-grid screening. It is a much more plausible unified
physical reward than previous candidates and no longer shows pronking collapse.

Do not train PPO yet. The next gate is held-out top-k config validation for
the v4 training-range fair grid. This checks whether the selected best configs
and gait rankings survive fresh random seeds rather than only the search seed.
```

### 2026-06-21 v4 held-out config selection

Run:

```text
runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3
```

Input:

```text
runs/high_level_oracle_gait/fair_target_gait_audit/20260621_v4_physical_training_range_action_grid/fair_gait_grid_results.csv
```

Selection protocol:

```text
top_k_per_task_speed_gait = 3
score_keys:
  canonical_efficiency_v4_physical_score
  weighted_metric_reward_mean
  neutral_score

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

By task:

```text
flat_trot_efficiency: 73
push_lateral_pace_recovery: 46
ramp_up_trot_robustness: 73
rough_slope_trot_robustness: 69
stepping_stones_easy_bound_highspeed: 38
```

Decision:

```text
The selected held-out request set is balanced enough across gait families and
covers both v4 score-best and neutral-score alternatives. Use it for the next
fresh-seed validation pass.
```

Next input file:

```text
runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3/new_heldout_config_requests.csv
```

### 2026-06-21 v4 held-out validation seed208

Run:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208
```

Analysis:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed208/analysis
```

Protocol:

```text
config_csv:
  runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3/new_heldout_config_requests.csv

configs = 299
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
repeats_per_config = 4
steps = 500
warmup_steps = 100
seed = 208
```

Best-config winner counts:

```text
trotting: 9
pronking: 6
pacing: 2
bounding: 0
```

Best-config winners:

```text
flat_trot_efficiency:
  vx=0.5 -> trotting, score 0.9256, vx_err 0.0657
  vx=1.0 -> trotting, score 0.8775, vx_err 0.1110
  vx=1.5 -> trotting, score 0.8146, vx_err 0.2162
  vx=2.0 -> trotting, score 0.7505, vx_err 0.3526

push_lateral_pace_recovery:
  vx=1.2 -> trotting, score 0.8105, vx_err 0.2051
  vx=1.5 -> trotting, score 0.7884, vx_err 0.2492
  vx=1.8 -> trotting, score 0.7441, vx_err 0.3561

ramp_up_trot_robustness:
  vx=0.5 -> pronking, score 0.9122, vx_err 0.0736
  vx=1.0 -> pronking, score 0.8694, vx_err 0.1246
  vx=1.5 -> pronking, score 0.8012, vx_err 0.2407
  vx=2.0 -> pronking, score 0.7006, vx_err 0.4451

rough_slope_trot_robustness:
  vx=0.5 -> pronking, score 0.8943, vx_err 0.0884
  vx=1.0 -> trotting, score 0.8267, vx_err 0.1675
  vx=1.5 -> pronking, score 0.7406, vx_err 0.3224
  vx=2.0 -> trotting, score 0.6117, vx_err 0.5843

stepping_stones_easy_bound_highspeed:
  vx=1.7 -> pacing, score 0.6454, vx_err 0.4445
  vx=2.0 -> pacing, score 0.5910, vx_err 0.5852
```

Search seed207 vs held-out seed208:

```text
unchanged task-speed winners: 14 / 17

changed:
  ramp 2.0:   trotting -> pronking
  rough 1.0:  pronking -> trotting
  stones 1.7: trotting -> pacing
```

Top-3-config mean winners on seed208:

```text
trotting: 8
pronking: 7
pacing: 2
```

Top-3 mean differs from best-config winner at:

```text
rough 1.0:
  best config -> trotting
  top-3 mean  -> pronking
```

Interpretation:

```text
The held-out seed208 result mostly supports v4. It is not a max-over-grid
artifact in the old sense: the broad structure remains stable, with flat and
push consistently trotting, ramp mostly pronking, rough mixed by speed, and
stones pacing/mixed with weak task quality.

The unstable points are exactly the regions already flagged as weak:
  ramp 2.0, rough 1.0, and stones 1.7.

rough 1.0 should be treated as uncertain rather than a stable trot or pronk
winner, because best-config and top-3-mean disagree.

stones remains quality-limited because vx_err is still high, even when pacing
wins.
```

Decision:

```text
Do not promote v4 to PPO from a single held-out seed.
Run one more held-out validation seed using the same 299-config request set.
If the second held-out seed preserves the same broad structure and does not
expose a new collapse or compensation pathology, v4 can move to a short PPO
diagnostic run.
```

### 2026-06-22 v4 held-out validation seed209

Run:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209
```

Analysis:

```text
runs/high_level_oracle_gait/heldout_validation/20260621_v4_training_range_topk_k3_seed209/analysis
```

Protocol:

```text
config_csv:
  runs/high_level_oracle_gait/heldout_config_selection/20260621_v4_training_range_topk_k3/new_heldout_config_requests.csv

configs = 299
reward_profile = canonical_efficiency_v4_physical
selection_score_key = weighted_metric_reward_mean
repeats_per_config = 4
steps = 500
warmup_steps = 100
seed = 209
```

Best-config winner counts:

```text
trotting: 9
pronking: 7
pacing: 1
bounding: 0
```

Best-config winners:

```text
flat_trot_efficiency:
  vx=0.5 -> trotting, score 0.9282, vx_err 0.0554
  vx=1.0 -> trotting, score 0.8760, vx_err 0.1157
  vx=1.5 -> trotting, score 0.8180, vx_err 0.2038
  vx=2.0 -> trotting, score 0.7464, vx_err 0.3587

push_lateral_pace_recovery:
  vx=1.2 -> trotting, score 0.8169, vx_err 0.2047
  vx=1.5 -> trotting, score 0.7925, vx_err 0.2482
  vx=1.8 -> trotting, score 0.7419, vx_err 0.3776

ramp_up_trot_robustness:
  vx=0.5 -> pronking, score 0.9097, vx_err 0.0702
  vx=1.0 -> pronking, score 0.8666, vx_err 0.1324
  vx=1.5 -> pronking, score 0.8036, vx_err 0.2389
  vx=2.0 -> pronking, score 0.7024, vx_err 0.4397

rough_slope_trot_robustness:
  vx=0.5 -> trotting, score 0.8927, vx_err 0.0844
  vx=1.0 -> pronking, score 0.8257, vx_err 0.1825
  vx=1.5 -> pronking, score 0.7455, vx_err 0.3207
  vx=2.0 -> trotting, score 0.6093, vx_err 0.5994

stepping_stones_easy_bound_highspeed:
  vx=1.7 -> pronking, score 0.6408, vx_err 0.5061
  vx=2.0 -> pacing, score 0.5883, vx_err 0.5909
```

Three-run stability:

```text
search seed207 + held-out seed208 + held-out seed209:

same winner in all three runs: 13 / 17 task-speed points

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
  search seed207 says trotting.
  held-out seeds 208 and 209 both say pronking.
  The seed209 pronking win is small and comes from contact/yaw/lateral/impact,
  while trotting still has better progress and gated power.

rough 0.5:
  search seed207 and held-out seed208 say pronking.
  held-out seed209 says trotting by only about 0.0012.
  This is a near tie.

rough 1.0:
  search seed207 and held-out seed209 best-config winner is pronking.
  held-out seed208 best-config winner is trotting.
  Top-3 mean also disagrees across seeds, so this remains uncertain.

stones 1.7:
  search seed207 says trotting.
  held-out seed208 says pacing.
  held-out seed209 best-config winner says pronking, but seed209 top-3 mean
  says pacing.
  This point remains task-quality weak and should not become a hard gait target.
```

Top-3-config mean on seed209:

```text
trotting: 10
pronking: 5
pacing: 2
```

Top-3 mean differs from best-config winner at:

```text
rough 1.0:
  best config -> pronking
  top-3 mean  -> trotting

stones 1.7:
  best config -> pronking
  top-3 mean  -> pacing
```

Decision:

```text
v4 passes the second held-out screen at the broad-structure level. The stable
structure is:

  flat: consistently trotting
  push: consistently trotting
  ramp: mostly pronking, with 2.0 shifting to pronking in held-out
  rough: mixed and speed-dependent
  stones: pacing at 2.0, unstable/weak at 1.7

This is strong enough to justify a short PPO diagnostic run, but not enough to
claim final reward validity. Keep canonical_efficiency_v4_physical marked
diagnostic-only and use `--allow-diagnostic-reward-profile` deliberately.
```

### 2026-06-22 short PPO diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100
```

Analysis:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100/analysis
```

Protocol:

```text
reward_profile = canonical_efficiency_v4_physical
allow_diagnostic_reward_profile = true
oracle_condition_obs = false
style_reward_scale = 0.0
z_dim = 16
adaptation_coef = 0.1
selector_only = false
selector_hold_steps = 3
iterations = 100
num_envs = 256
num_steps = 32
```

Training-log summary:

```text
weighted_metric_reward:
  early 0.6564 -> late 0.6764
  best row 0.6860

vx_err:
  early 0.4456 -> late 0.4201
  best row 0.3940

score_progress:
  early 0.6076 -> late 0.6386

score_strict_gated_power_efficiency:
  early 0.1885 -> late 0.2142

done_rate:
  early 0.0176 -> late 0.0189

gait_switch_rate:
  early 0.2355 -> late 0.2346

action_clip_rate:
  late about 0.00001

z_error:
  iter0 0.0198 -> iter99 0.0024
```

Final mixed-rollout gait ratios:

```text
overall:
  pronking 0.316
  trotting 0.341
  bounding 0.204
  pacing 0.140

flat_trot_efficiency:
  pronking 0.302
  trotting 0.345
  bounding 0.205
  pacing 0.148

ramp_up_trot_robustness:
  pronking 0.306
  trotting 0.348
  bounding 0.211
  pacing 0.135

rough_slope_trot_robustness:
  pronking 0.328
  trotting 0.350
  bounding 0.184
  pacing 0.138

push_lateral_pace_recovery:
  pronking 0.317
  trotting 0.330
  bounding 0.205
  pacing 0.148

stepping_stones_easy_bound_highspeed:
  pronking 0.327
  trotting 0.329
  bounding 0.214
  pacing 0.129
```

Continuous action health:

```text
frequency_mean:
  iter0 2.855 -> iter99 2.967

footswing_height_mean:
  iter0 0.0935 -> iter99 0.0896

stance_width_mean:
  iter0 0.3481 -> iter99 0.3326

body_pitch_mean:
  iter0 0.0028 -> iter99 0.0076

action_clip_rate:
  effectively zero
```

Interpretation:

```text
The short PPO diagnostic is stable and learns a little: v4 reward, progress,
vx_err, and gated power improve, and the RMA student latent converges toward the
teacher latent. There is no action-clipping failure.

However, the selector has not learned clear condition-specific gait separation
in the mixed training rollout. All tasks end with very similar gait ratios,
roughly trot/pronk-heavy with residual bounding/pacing. This does not match the
fair-audit expectation of flat/push trotting, ramp pronking, rough mixed, and
stones pacing/mixed.

The result should be interpreted as:
  v4 is learnable as a performance reward;
  no-task RMA PPO for 100 iterations has not yet produced visible gait
  condition differentiation;
  independent per-task checkpoint evaluation is required before deciding
  whether to continue training, switch to oracle one-hot ablation, or adjust
  selector credit assignment.
```

Decision:

```text
Do not start a long PPO run yet.
Next run independent per-task evaluation on checkpoint high_level_000099.pt.
Judge deterministic per-task gait ratios and task metrics, not only mixed
training rollout ratios.
```

### 2026-06-22 independent eval for short PPO diagnostic

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100/independent_eval/20260622_full_iter099
```

Checkpoint:

```text
runs/high_level_oracle_gait/20260622_v4_physical_notask_rma_iter100/checkpoints/high_level_000099.pt
```

Protocol:

```text
full independent per-task evaluation
num_envs = 32
steps = 1000
warmup_steps = 50
oracle_condition_obs = false
selector_only = false
```

Dominant gait per task-speed:

```text
flat_trot_efficiency:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> pronking by a tiny margin

ramp_up_trot_robustness:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> trotting

rough_slope_trot_robustness:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> trotting

push_lateral_pace_recovery:
  vx=1.5 -> trotting

stepping_stones_easy_bound_highspeed:
  vx=2.0 -> trotting
```

Average gait ratios by task:

```text
flat:
  pronking 0.384
  trotting 0.421
  bounding 0.132
  pacing 0.063

ramp:
  pronking 0.376
  trotting 0.423
  bounding 0.128
  pacing 0.073

rough:
  pronking 0.369
  trotting 0.425
  bounding 0.132
  pacing 0.074

push:
  pronking 0.366
  trotting 0.394
  bounding 0.158
  pacing 0.082

stones:
  pronking 0.356
  trotting 0.378
  bounding 0.163
  pacing 0.103
```

Task-quality warning:

```text
vx_err remains high in hard/high-speed cases:
  ramp 2.0:   0.597
  rough 2.0:  0.820
  push 1.5:   0.511
  stones 2.0: 0.992
```

Interpretation:

```text
Independent evaluation confirms the mixed-rollout concern. The learned no-task
RMA policy is mostly a global trot/pronk mixture. Its gait ratios are very
similar across all tasks, so the selector has not learned the fair-audit
condition structure:

  expected from v4 fair/held-out audits:
    flat/push -> mostly trotting
    ramp -> mostly pronking
    rough -> mixed and speed-dependent
    stones -> pacing/mixed but weak

  learned no-task RMA checkpoint:
    almost all task-speed points -> trotting-dominant with substantial pronking
```

Decision:

```text
Do not continue directly to a long no-task RMA run.

The next diagnostic is an oracle task-onehot run with the same v4 reward and no
style/gait prior. This tests whether selector differentiation appears when task
identity is easy. If task-onehot v4 still does not differentiate, the bottleneck
is selector credit assignment or reward/action coupling, not RMA inference.
```

### 2026-06-22 oracle task-onehot v4 PPO diagnostic

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100
```

Analysis:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/analysis
```

Protocol:

```text
reward_profile = canonical_efficiency_v4_physical
allow_diagnostic_reward_profile = true
oracle_condition_obs = true
style_reward_scale = 0.0
z_dim = 16
adaptation_coef = 0.1
selector_only = false
selector_hold_steps = 3
iterations = 100
num_envs = 256
num_steps = 32
```

Training-log summary:

```text
weighted_metric_reward:
  early 0.6621 -> late 0.6770

vx_err:
  early 0.4314 -> late 0.4143

score_progress:
  early 0.6193 -> late 0.6432

score_strict_gated_power_efficiency:
  early 0.1963 -> late 0.2173

done_rate:
  early 0.0176 -> late 0.0188

gait_switch_rate:
  early 0.2359 -> late 0.2360

action_clip_rate:
  late 0.0000

z_error:
  early 0.0081 -> late 0.0032
```

Final mixed-rollout gait ratios:

```text
overall:
  pronking 0.228
  trotting 0.363
  bounding 0.258
  pacing 0.150

flat:
  pronking 0.208
  trotting 0.344
  bounding 0.291
  pacing 0.156

ramp:
  pronking 0.229
  trotting 0.375
  bounding 0.232
  pacing 0.165

rough:
  pronking 0.244
  trotting 0.347
  bounding 0.257
  pacing 0.152

push:
  pronking 0.231
  trotting 0.386
  bounding 0.251
  pacing 0.132

stones:
  pronking 0.230
  trotting 0.365
  bounding 0.260
  pacing 0.145
```

Interpretation:

```text
The task-onehot run is stable and learns the v4 performance reward slightly.
Compared with the no-task RMA run, the mixed rollout shifts away from the
previous trot/pronk mixture toward a trot/bound-heavy mixture. But it still does
not show clean condition-driven gait differentiation: push remains trotting
dominant rather than pacing, and stones remains trotting/bounding mixed rather
than a clean pacing/bounding solution.

Mixed rollout is not enough to judge the checkpoint. The required next step is
deterministic independent per-task evaluation of checkpoint high_level_000099.pt.
If independent eval also shows no task-conditioned differentiation, then
task-onehot did not solve selector credit assignment, and direct longer PPO is
not justified.
```

### 2026-06-22 independent eval for oracle task-onehot v4 diagnostic

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/independent_eval/20260622_full_iter099
```

Checkpoint:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_iter100/checkpoints/high_level_000099.pt
```

Protocol:

```text
full independent per-task evaluation
num_envs = 32
steps = 1000
warmup_steps = 50
oracle_condition_obs = true
selector_only = false
```

Dominant gait per task-speed:

```text
flat_trot_efficiency:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> trotting

ramp_up_trot_robustness:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> trotting

rough_slope_trot_robustness:
  vx=0.5 -> trotting
  vx=1.0 -> trotting
  vx=1.5 -> trotting
  vx=2.0 -> trotting

push_lateral_pace_recovery:
  vx=1.5 -> trotting

stepping_stones_easy_bound_highspeed:
  vx=2.0 -> trotting
```

Average gait ratios by task:

```text
flat:
  pronking 0.277
  trotting 0.450
  bounding 0.252
  pacing 0.020

ramp:
  pronking 0.271
  trotting 0.457
  bounding 0.245
  pacing 0.026

rough:
  pronking 0.249
  trotting 0.462
  bounding 0.259
  pacing 0.030

push:
  pronking 0.253
  trotting 0.442
  bounding 0.274
  pacing 0.032

stones:
  pronking 0.213
  trotting 0.450
  bounding 0.289
  pacing 0.048
```

Overall average gait ratios:

```text
pronking 0.261
trotting 0.455
bounding 0.256
pacing 0.028
```

Task-quality warning:

```text
vx_err remains high at difficult/high-speed points:
  flat 2.0:   0.532
  ramp 2.0:   0.602
  rough 2.0:  0.804
  push 1.5:   0.500
  stones 2.0: 1.012
```

Interpretation:

```text
Independent evaluation answers the task-onehot diagnostic: providing oracle
task identity still does not make the learned selector follow the v4 fair-audit
condition structure. Instead, it produces an even clearer global trotting
dominance, with bounding as a broad secondary mode and almost no pacing.

This means the immediate bottleneck is not just RMA/no-task inference. The
current v4 PPO setup can improve performance reward while ignoring the
fair-audit gait ranking. The likely issue is selector credit assignment and/or
coupling between the discrete gait selector and continuous residuals.
```

Decision:

```text
Do not continue direct long training of the current v4 full-action PPO setup.

Next should be a mechanism diagnostic rather than another long run:
  1. selector-only + task-onehot under canonical_efficiency_v4_physical;
  2. continuous-only/fixed-gait or fixed-selector controls;
  3. compare whether continuous residuals can absorb most v4 reward improvement;
  4. if selector-only still does not follow fair-audit ranking, consider explicit
     score-derived selector supervision or a different discrete-action credit
     assignment mechanism as an ablation, not as the main unified-reward claim.
```

### 2026-06-22 selector-only + task-onehot v4 PPO diagnostic

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100
```

Analysis:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/analysis
```

Protocol:

```text
reward_profile = canonical_efficiency_v4_physical
allow_diagnostic_reward_profile = true
oracle_condition_obs = true
selector_only = true
style_reward_scale = 0.0
z_dim = 16
adaptation_coef = 0.1
selector_hold_steps = 3
iterations = 100
num_envs = 256
num_steps = 32
continuous residuals executed as zero
```

Training-log summary:

```text
weighted_metric_reward:
  early 0.6626 -> late 0.6672

vx_err:
  early 0.4297 -> late 0.4267

score_progress:
  early 0.6194 -> late 0.6265

score_strict_gated_power_efficiency:
  early 0.1955 -> late 0.1994

done_rate:
  early 0.0174 -> late 0.0187

gait_switch_rate:
  early 0.2363 -> late 0.2369

action_clip_rate:
  late 0.0000

z_error:
  early 0.0074 -> late 0.0021
```

Final mixed-rollout gait ratios:

```text
overall:
  pronking 0.261
  trotting 0.251
  bounding 0.252
  pacing 0.236

flat:
  pronking 0.273
  trotting 0.246
  bounding 0.225
  pacing 0.256

ramp:
  pronking 0.263
  trotting 0.233
  bounding 0.267
  pacing 0.237

rough:
  pronking 0.263
  trotting 0.282
  bounding 0.241
  pacing 0.213

push:
  pronking 0.249
  trotting 0.251
  bounding 0.257
  pacing 0.242

stones:
  pronking 0.253
  trotting 0.245
  bounding 0.268
  pacing 0.234
```

Interpretation:

```text
Selector-only training with task-onehot gives an even clearer mechanism signal:
the v4 reward provides only a weak gradient to the discrete gait selector. With
continuous residuals fixed at zero, reward and progress improve only slightly,
and the gait distribution remains close to uniform rather than following the
v4 fair-audit structure.

This weakens the hypothesis that continuous residuals alone were hiding a good
selector signal. It points more directly to selector credit assignment under
reward-only PPO.
```

Decision:

```text
Run independent eval once for completeness. If independent eval also remains
near-uniform, stop reward-only PPO diagnostics for v4 and design the next
ablation around explicit selector credit:
  - score-derived soft selector target from fair/held-out audits;
  - or per-gait candidate evaluation / bandit-style selector credit;
  - while keeping unified physical reward as the main performance objective.
```

### 2026-06-22 independent eval for selector-only + task-onehot v4 diagnostic

Run:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/independent_eval/20260622_full_iter099
```

Checkpoint:

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_selector_only_iter100/checkpoints/high_level_000099.pt
```

Protocol:

```text
full independent per-task evaluation
num_envs = 32
steps = 1000
warmup_steps = 50
oracle_condition_obs = true
selector_only = true
continuous residuals executed as zero
```

Dominant gait per task-speed:

```text
flat_trot_efficiency:
  vx=0.5 -> pronking
  vx=1.0 -> pronking
  vx=1.5 -> pronking
  vx=2.0 -> pronking

ramp_up_trot_robustness:
  vx=0.5 -> pronking
  vx=1.0 -> pronking
  vx=1.5 -> pronking
  vx=2.0 -> pronking

rough_slope_trot_robustness:
  vx=0.5 -> pronking
  vx=1.0 -> pronking
  vx=1.5 -> pronking
  vx=2.0 -> pronking

push_lateral_pace_recovery:
  vx=1.5 -> pronking

stepping_stones_easy_bound_highspeed:
  vx=2.0 -> pronking
```

Average gait ratios by task:

```text
flat:
  pronking 0.294
  trotting 0.263
  bounding 0.231
  pacing 0.212

ramp:
  pronking 0.289
  trotting 0.259
  bounding 0.229
  pacing 0.223

rough:
  pronking 0.286
  trotting 0.253
  bounding 0.239
  pacing 0.223

push:
  pronking 0.300
  trotting 0.247
  bounding 0.227
  pacing 0.226

stones:
  pronking 0.276
  trotting 0.234
  bounding 0.254
  pacing 0.236
```

Overall average gait ratios:

```text
pronking 0.289
trotting 0.256
bounding 0.234
pacing 0.221
```

Task-quality warning:

```text
vx_err remains high at difficult/high-speed points:
  flat 2.0:   0.581
  ramp 2.0:   0.656
  rough 2.0:  0.830
  push 1.5:   0.529
  stones 2.0: 1.002
```

Interpretation:

```text
The selector-only independent evaluation confirms that canonical_efficiency_v4_physical
does not provide a usable condition-specific discrete-selector signal under
reward-only PPO. Even with task-onehot and zero continuous residuals, the policy
does not follow the fair/held-out gait structure. It only develops a weak global
pronk preference while remaining close to uniform.

This also weakens the earlier hypothesis that continuous residuals were the
main reason the selector ignored fair-audit rankings. The deeper issue is
credit assignment to the discrete selector from the unified performance reward.
```

Decision:

```text
Stop direct reward-only PPO diagnostics for v4. Do not run longer versions of:
  - no-task RMA full-action v4,
  - task-onehot full-action v4,
  - task-onehot selector-only v4.

The next step should be an explicit selector-credit ablation:
  1. build score-derived soft selector targets from the v4 fair-grid and
     held-out validation results, conditioned on task and speed;
  2. train with unified physical reward plus a small selector KL / CE auxiliary;
  3. keep this as an ablation/diagnostic, not the main unified-reward claim;
  4. evaluate whether the selector can follow the fair-audit structure when the
     credit assignment problem is made easier.
```

## 2026-06-22: Implemented Selector Reference Diagnostic

中文说明：

```text
我们不再继续指望“只靠统一物理奖励”自然产生清晰的步态分化。

现在新增一个诊断实验：
  用已经复测过的 v4 结果，生成一张表：
    场景 + 速度 -> 四种步态的参考概率

训练时，这张表只给“步态选择输出”一个很小的参考约束。
它不改变环境里的物理奖励，也不直接要求连续参数变成某个值。
```

实现文件：

```text
scripts/build_soft_selector_targets.py
scripts/train_high_level_oracle_ppo.py
```

新增训练参数：

```text
--selector-targets
  参考概率表路径。

--selector-aux-coef
  参考训练项的系数。为 0 时完全关闭。

--selector-aux-min-confidence
  低于该可信度的参考行不参与训练。
```

已生成参考表：

```text
runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv
```

参考表统计：

```text
rows: 17
pronking top: 6
trotting top: 9
bounding top: 0
pacing top: 2
confidence < 0.25: 10
```

下一步实验顺序：

```text
1. 直接告诉网络任务 + 只允许选择步态 + 小参考训练项。
   目的：先确认步态选择输出和参考表训练项能不能工作。

2. 如果第 1 步能按场景/速度分化，再允许连续参数参与。
   目的：检查连续参数打开后，步态分化是否还能保住。

3. 最后再尝试不直接告诉任务，只靠本体历史学习。
   目的：检查真实部署形式是否能从历史状态中推断场景差异。
```

## 2026-06-23: First Selector Reference Run Finished

已完成：

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_choose_gait_refprob_coef005_iter100
```

实验含义：

```text
直接告诉网络当前任务；
只训练步态选择，连续参数固定为 0；
在统一物理奖励之外，对步态选择输出加入很小的参考训练项。
```

训练日志结果：

```text
weighted_metric_reward: 0.6463 -> 0.6820
vx_err:                 0.4450 -> 0.4055
参考训练项损失:          1.4056 -> 1.3180
步态输出熵:              1.3433 -> 1.1820

final mixed gait ratio:
  pronking 0.330
  trotting 0.345
  bounding 0.153
  pacing 0.172
```

按混合训练日志初步看：

```text
参考训练项确实接上了，步态输出不再完全接近平均；
但 0.05 系数还没有让所有场景明确贴合参考表。

flat:   pronk/trot 近似并列
ramp:   偏 pronk
rough:  pronk/trot 接近
push:   偏 trot
stones: 没有偏 pace，仍偏 trot
```

下一步：

```text
先跑独立评测，固定每个场景和速度逐项检查。
不要只根据混合训练日志决定是否加大系数。
```

独立评测结果：

```text
runs/high_level_oracle_gait/20260622_v4_physical_taskonehot_choose_gait_refprob_coef005_iter100/independent_eval/20260623_training_range_iter099

固定 17 个训练速度点：
  参考表第一步态 vs 实际第一步态 = 11 / 17 匹配
  按参考表可信度加权的匹配度 = 0.879
```

关键拆分：

```text
可信度 >= 0.5 的参考点：
  7 / 7 匹配

可信度 < 0.25 的参考点：
  4 / 10 匹配
```

实际第一步态统计：

```text
pronking 6
trotting 11
bounding 0
pacing 0
```

判断：

```text
这说明参考训练项本身已经有效。
在参考表可信度高的点，网络能按场景/速度改变步态选择。

失败主要集中在参考表本来就弱的点，尤其 stones。
stones 的速度跟踪质量很差，不能靠强行加大参考系数来解释为合理分化。
```

下一步：

```text
进入“直接告诉任务 + 可以选择步态也可以调连续参数 + 小参考训练项”。

目的：
  检查连续参数打开后，高可信度点的步态结构是否还能保住；
  同时看运动性能是否比只选步态更好。

暂时不要做不告诉任务的版本。
```

## 2026-06-23: Full-Action Selector Reference Run Finished

已完成：

```text
runs/high_level_oracle_gait/20260623_v4_physical_taskonehot_fullaction_refprob_coef005_iter100
```

实验含义：

```text
直接告诉网络当前任务；
允许选择步态，也允许调连续参数；
在统一物理奖励之外，对步态选择输出加入同一张参考概率表。
```

训练日志结果：

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

和“只允许选择步态”的上一轮对比：

```text
只允许选择步态：
  reward = 0.6820
  vx_err = 0.4055

允许连续参数：
  reward = 0.6747
  vx_err = 0.4281
```

初步判断：

```text
打开连续参数后，训练日志没有显示运动表现提升。
步态比例更偏向全局 trot，坡地/粗糙地该偏 pronk 的结构变弱。
连续参数均值在不同任务之间差异很小，目前不像学出了有效的分场景调节。
```

下一步：

```text
必须跑固定 17 个训练速度点的独立评测。
如果独立评测也显示高可信度点结构被冲掉，就先不要进入“不告诉任务”的版本。
```

独立评测结果：

```text
runs/high_level_oracle_gait/20260623_v4_physical_taskonehot_fullaction_refprob_coef005_iter100/independent_eval/20260623_training_range_iter099
```

固定 17 个训练速度点：

```text
参考表第一步态 vs 实际第一步态 = 9 / 17 匹配
按参考表可信度加权的匹配度 = 0.649

实际第一步态：
  trotting = 17 / 17
```

与“只允许选择步态”的版本相比：

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

指标对比：

```text
full-action 的速度误差略低、步态切换更少；
但统一物理奖励均值更低，且步态结构塌到全局 trot。
```

判断：

```text
selector_aux_coef=0.05 对 full-action 太弱。
连续参数打开后，策略选择了更稳定的全局 trot 方案，
没有保住 ramp 等高可信度点的 pronk 结构。
```

下一步：

```text
不要进入不告诉任务/RMA 版本。
先跑 full-action + task-onehot + 只使用高可信度参考行 + 更大参考系数。

建议：
  --selector-aux-min-confidence 0.25
  --selector-aux-coef 0.15

目的：
  不再强行监督 stones 等低可信度点；
  只检查高可信度点，尤其 ramp 1.0 / ramp 1.5 的 pronk，
  能不能在连续参数打开后保住。
```

## 2026-06-24: High-Confidence Selector Reference Full-Action Run Finished

已完成：

```text
runs/high_level_oracle_gait/20260624_v4_physical_taskonehot_fullaction_refprob_highconf_coef015_iter100
```

这次实验做的事情：

```text
直接告诉网络任务编号；
允许网络同时选择步态和调连续参数；
只对参考表中可信度 >= 0.25 的场景/速度给步态参考训练项；
把步态参考训练项系数提高到 0.15。
```

为什么这样做：

```text
上一轮完整动作版本虽然运动略稳定，但固定速度独立评测中变成 17/17
全是 trotting，说明 0.05 的步态参考训练项太弱。

这次不再强行监督低可信度的 stones 等点，只看参考表较可靠的位置能不能
在连续参数打开后保住步态结构。
```

训练日志结果：

```text
weighted_metric_reward: 0.6293 -> 0.6794
vx_err:                 0.4873 -> 0.4100
步态参考训练项损失:      1.3801 -> 1.2000
步态参考训练项平均权重:  0.3320 -> 0.3033
参考表熵:                1.1651 -> 1.1678
步态输出熵:              1.3477 -> 1.1818

最终混合训练步态比例：
  pronking 0.347
  trotting 0.348
  bounding 0.131
  pacing 0.174
```

按任务拆分：

```text
flat:
  pronking 0.344, trotting 0.334, bounding 0.130, pacing 0.192

ramp:
  pronking 0.474, trotting 0.265, bounding 0.124, pacing 0.137

rough:
  pronking 0.352, trotting 0.351, bounding 0.120, pacing 0.177

push:
  pronking 0.250, trotting 0.425, bounding 0.136, pacing 0.189

stones:
  pronking 0.316, trotting 0.365, bounding 0.148, pacing 0.172
```

连续参数：

```text
整体均值：
  frequency        2.9207
  duration         0.5189
  footswing_height 0.0919
  stance_width     0.3496
  body_pitch       0.0007
  action_clip_rate 0.0

不同任务之间仍然差异很小，尚未证明连续参数学出了清楚的分场景调节。
```

判断：

```text
这次混合训练日志比 coefficient=0.05 的完整动作版本更好：
  ramp 明显更偏 pronking；
  push 更偏 trotting；
  rough 接近 pronking/trotting 并列；
  overall 不再明显塌到全局 trotting。

但这仍然不能当作最终结论，因为混合训练日志会被任务采样、速度分布和
短期随机性影响。必须用固定任务/固定速度独立评测来确认。
```

下一步：

```text
对该 run 跑固定 17 个训练速度点的独立评测。

评测后重点看：
  1. 参考表高可信度点是否匹配；
  2. 按参考表可信度加权后的匹配度是否高于 coefficient=0.05 完整动作版本；
  3. ramp 1.0 / ramp 1.5 是否保住 pronking；
  4. flat / push 是否保住 trotting；
  5. stones 等低可信度点不要强行解读；
  6. 连续参数是否出现任务/速度相关差异。
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

低可信度失败点：

```text
rough 1.0:
  reference trotting, actual pronking, confidence 0.019

rough 1.5:
  reference pronking, actual trotting, confidence 0.155

stones 1.7:
  reference pacing, actual trotting, confidence 0.183
  vx_err = 0.710

stones 2.0:
  reference pacing, actual trotting, confidence 0.102
  vx_err = 0.902
```

与前两轮固定速度评测对比：

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
这次结果确认：在直接告诉任务编号的条件下，步态参考训练项已经有效。
它解决了 coefficient=0.05 完整动作版本塌到全局 trotting 的问题，
并且没有破坏速度跟踪；平均 vx_err 反而是三轮里最低的。

当前失败集中在低可信度行，尤其 stones。stones 的速度误差很高，所以不要
把 stones 没有学成 pacing 当作强负面结论。

bounding 仍没有出现，但当前参考表本身没有任何 task-speed 点以 bounding
为第一步态，因此这不是本轮诊断要解决的问题。
```

下一步：

```text
进入不直接告诉任务编号的 RMA 版本。

目的：
  检查网络能否从本体历史里识别场景/速度差异，并复现这次在“直接告诉任务”
  条件下已经证明可行的步态结构。

如果 no-task/RMA 失败：
  问题就更可能在场景识别/RMA 表征，而不是步态参考训练项本身。

如果 no-task/RMA 成功：
  就说明可以在不直接给任务编号的情况下，通过本体历史和参考训练项学出
  条件相关的步态选择。
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
允许网络同时选择步态和调连续参数；
只对参考表中可信度 >= 0.25 的场景/速度给步态参考训练项；
步态参考训练项系数保持 0.15。
```

为什么这样做：

```text
上一轮已经证明：如果直接告诉网络任务编号，步态参考训练项可以有效工作。
这次要检查的是：不直接告诉任务编号时，网络能不能从本体历史里识别场景，
并复现类似的条件相关步态结构。
```

训练日志结果：

```text
weighted_metric_reward: 0.6259 -> 0.6903
vx_err:                 0.4940 -> 0.3996
步态参考训练项损失:      1.3789 -> 1.2309
步态参考训练项平均权重:  0.3420 -> 0.3238
步态输出熵:              1.3525 -> 1.2156
RMA 蒸馏误差:            0.0180 -> 0.0025

最终混合训练步态比例：
  pronking 0.326
  trotting 0.384
  bounding 0.134
  pacing 0.156
```

按任务拆分：

```text
flat:
  pronking 0.326, trotting 0.380, bounding 0.132, pacing 0.162

ramp:
  pronking 0.360, trotting 0.361, bounding 0.124, pacing 0.156

rough:
  pronking 0.368, trotting 0.370, bounding 0.126, pacing 0.135

push:
  pronking 0.290, trotting 0.401, bounding 0.153, pacing 0.156

stones:
  pronking 0.284, trotting 0.407, bounding 0.137, pacing 0.172
```

与“直接告诉任务编号”的同配置版本对比：

```text
直接告诉任务编号：
  weighted_metric_reward = 0.6794
  vx_err = 0.4100
  final ratios = pronking 0.347, trotting 0.348, bounding 0.131, pacing 0.174
  mixed log 中 ramp 明显偏 pronking

不直接告诉任务编号，用本体历史/RMA：
  weighted_metric_reward = 0.6903
  vx_err = 0.3996
  final ratios = pronking 0.326, trotting 0.384, bounding 0.134, pacing 0.156
  mixed log 中五类任务最终都略偏 trotting
```

初步判断：

```text
这次不是简单失败。它的运动指标更好，RMA 蒸馏也正常收敛。

但是，从混合训练日志看，步态结构比“直接告诉任务编号”的版本弱。
最关键的是 ramp 没有明显保住 pronking，而是和 trotting 接近甚至略偏
trotting。

因此现在不能下结论。必须跑固定 17 个训练速度点的独立评测。
```

下一步：

```text
对该 run 跑固定 17 个训练速度点的独立评测。

评测后重点看：
  1. 高可信度参考点是否仍然 7 / 7 匹配；
  2. ramp 1.0 / ramp 1.5 是否保住 pronking；
  3. flat 1.5 / flat 2.0 和 push 1.2 / 1.5 / 1.8 是否保住 trotting；
  4. 按参考表可信度加权后的匹配度是否接近直接任务版本的 0.932；
  5. 如果匹配明显下降，问题更可能是场景识别/RMA 表征，而不是步态参考训练项。
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
flat 1.5:
  reference trotting, actual trotting

flat 2.0:
  reference trotting, actual trotting

ramp 1.0:
  reference pronking, actual trotting

ramp 1.5:
  reference pronking, actual trotting

push 1.2:
  reference trotting, actual trotting

push 1.5:
  reference trotting, actual trotting

push 1.8:
  reference trotting, actual trotting
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
这次不是运动性能失败。它的平均速度误差更低，步态切换更少。

但它是步态条件化失败：不直接告诉任务编号后，坡地 1.0 / 1.5 这两个
高可信度跳步点没有保住，实际变成小跑。

因此当前问题更像是：
  本体历史/RMA 表征没有把场景差异有效传给步态选择输出；
  或者同样 0.15 的步态参考训练项，在没有任务编号时强度不够。

这不否定参考表，因为直接告诉任务编号的版本已经证明参考表有效。
```

下一步候选：

```text
先做诊断，不要直接堆更复杂训练。

建议顺序：
  1. 检查 RMA 隐变量是否能区分 flat/ramp/rough/push/stones；
  2. 对比 ramp 1.0 / ramp 1.5 上两个版本的步态输出概率；
  3. 如果隐变量分不开场景，优先改历史表征或蒸馏；
  4. 如果隐变量能分开但步态输出仍不按参考走，再尝试更强步态参考训练项
     或分阶段训练。
```

## 2026-06-26: Information-Path Diagnostic Becomes the Next Main Step

当前判断：

```text
不要继续把主要精力放在调统一物理奖励上。

canonical_efficiency_v4_physical 已经足够作为当前阶段的统一运动性能目标：
  - 它通过了在线/离线一致性检查；
  - 通过了小规模指标合理性检查；
  - 通过了训练速度范围的公平参数扫描和独立复测；
  - 短训后能改善运动指标。

但是，它不能单独让离散步态选择自然分化。
加入步态参考训练项后：
  - 如果直接告诉网络任务编号，高可信度参考点 7 / 7 匹配；
  - 如果不告诉任务编号、只用本体历史/RMA，高可信度参考点降到 5 / 7，
    主要失败在 ramp 1.0 / ramp 1.5。
```

因此，现在的瓶颈不应该简单写成“RMA 不行”。需要拆成四段信息通路：

```text
本体历史输入
  -> RMA 隐变量
  -> 步态选择输出
  -> 训练信号是否足够推动步态选择依赖这些信息
```

下一步先做诊断，不直接改网络、不直接长训：

```text
1. 采集固定评测数据：
   保存 task id、命令速度、本体历史输入、教师隐变量、学生隐变量、
   步态选择概率、实际选择步态、参考步态概率。

2. 检查本体历史输入有没有场景/速度信息：
   用简单诊断模型从本体历史预测任务类别、速度区间、参考第一步态。

3. 检查 RMA 隐变量有没有保留这些信息：
   分别用教师隐变量和学生隐变量预测任务类别、速度区间、参考第一步态。

4. 检查步态选择输出是否使用隐变量：
   对同一批样本替换、清零或扰动隐变量，看四种步态选择概率是否明显变化。

5. 专门对比 ramp 1.0 / ramp 1.5：
   这两个点在直接任务编号版本中能选择 pronking，
   在不直接告诉任务编号版本中变成 trotting，是当前最关键的失败案例。
```

诊断结果对应的决策：

```text
如果本体历史本身分不开场景：
  优先改输入内容或历史长度，例如加入更明确的接触历史、速度误差历史、
  姿态/角速度历史、足端滑移或接触力统计。

如果本体历史能分开，但学生隐变量分不开：
  优先改 RMA/历史压缩模块，例如增大隐变量维度、延长历史窗口、
  强化蒸馏目标，或让教师隐变量包含更有用的物理量。

如果隐变量能分开，但步态选择输出不随隐变量变化：
  说明策略没有使用这些信息。下一步考虑分阶段训练、
  更强的步态参考训练项，或者先限制连续参数、先训练步态选择。

如果步态选择输出会随隐变量变化，但仍不匹配参考：
  再检查参考训练项强度、可信度筛选、速度匹配方式和训练日程。
```

本阶段不要做的事：

```text
不要为了让 ramp 出现 pronking 继续改统一物理奖励；
不要直接上更长的不告诉任务编号训练；
不要把任务编号作为最终主线输入；
不要把步态参考训练项包装成自然涌现结果。
```

Implemented diagnostic tools:

```text
scripts/collect_high_level_info_path_data.py

作用：
  跑固定评测点，保存每个样本的：
    - 本体历史输入；
    - 教师隐变量；
    - 学生隐变量；
    - 四种步态选择概率；
    - 清零/打乱隐变量后的步态选择概率；
    - 参考步态概率；
    - 实际选择步态。

它回答：
  诊断需要的数据有没有被完整记录下来。
```

```text
scripts/analyze_high_level_info_path.py

作用：
  不再跑仿真，只读取上一个脚本保存的数据，做三类检查：
    - 本体历史能不能预测任务/速度/参考第一步态；
    - 教师隐变量、学生隐变量能不能预测这些信息；
    - 清零或打乱隐变量后，步态选择概率是否明显变化。

它回答：
  问题到底更像是“看不见”、 “压缩丢了”，还是“看见了但步态选择不用”。
```

Next command should collect diagnostic data from the no-task/RMA checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/collect_high_level_info_path_data.py \
  --run-dir runs/high_level_oracle_gait/20260625_v4_physical_notask_rma_fullaction_refprob_highconf_coef015_iter100 \
  --selector-targets runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv \
  --selector-aux-min-confidence 0.25 \
  --num-envs 32 \
  --samples-per-item 512 \
  --output-dir runs/high_level_oracle_gait/info_path_probe/20260626_notask_rma_highconf_coef015
```

Diagnostic data and analysis completed:

```text
runs/high_level_oracle_gait/info_path_probe/20260626_notask_rma_highconf_coef015

samples:
  17 task-speed points
  512 samples per point
  8704 samples total

analysis:
  runs/high_level_oracle_gait/info_path_probe/20260626_notask_rma_highconf_coef015/analysis
```

Simple diagnostic accuracy:

```text
history -> task:              0.553
history -> speed:             0.835
history -> reference top gait:0.830

student latent -> task:              0.537
student latent -> speed:             0.576
student latent -> reference top gait:0.916

teacher latent -> task:              0.941
teacher latent -> speed:             0.331
teacher latent -> reference top gait:1.000
```

Latent sensitivity of gait-selection probabilities:

```text
student vs teacher latent:
  mean_abs_prob_diff = 0.0088

student vs zero latent:
  mean_abs_prob_diff = 0.0077

student vs shuffled latent:
  mean_abs_prob_diff = 0.0051
```

Interpretation:

```text
The proprioceptive history is not empty: it predicts speed well and predicts
the high-confidence reference top gait reasonably well. It has some task
information, though task classification is only moderate.

The teacher latent strongly encodes task/reference information. The student
latent does not clearly encode task identity, but it can predict the
high-confidence reference top gait in this diagnostic.

The most important result is that gait-selection probabilities barely change
when the latent is replaced by the teacher latent, zeroed, or shuffled. This
means the current policy's gait-selection output is almost insensitive to the
RMA latent. The no-task/RMA failure is therefore primarily a "latent not used by
gait selection" problem, with a possible secondary student-latent/task-identity
weakness.
```

Key task-speed readout:

```text
ramp 1.0:
  reference = pronking
  student top = trotting
  pronking prob = 0.363
  trotting prob = 0.404

ramp 1.5:
  reference = pronking
  student top = trotting
  pronking prob = 0.295
  trotting prob = 0.457

flat 1.5 / 2.0 and push 1.2 / 1.5 / 1.8 still select trotting, matching the
high-confidence reference rows.
```

Updated next decision:

```text
Do not first change reward.
Do not first add task id as a final input.

The next training-design step should make the gait-selection output depend on
the condition latent before allowing the full policy to settle into a global
trot solution.

Candidate interventions:
  1. staged training: first train gait selection with latent/reference signal,
     then gradually unfreeze continuous residuals;
  2. stronger reference loss on the gait-selection output for high-confidence
     rows when task id is absent;
  3. latent-use regularization or latent dropout/swap consistency checks during
     training, so the gait selector cannot ignore the latent;
  4. if needed, improve student latent task separability after confirming that
     the selector actually uses it.
```

2026-06-26 decision after reviewing the diagnosis:

```text
The next minimal experiment should not be another reward edit and should not
immediately be a full-action run.

Run a z-conditioned gait-selector diagnostic:
  - no direct task id;
  - continuous residuals fixed at zero;
  - gait selection uses only student latent plus command speed;
  - high-confidence gait-reference rows supervise the gait-selection output;
  - v4 physical reward remains as the locomotion performance objective.
```

Purpose:

```text
This experiment asks a single question:

  If the gait selector is forced to make its decision from the RMA latent and
  command speed, can it recover the high-confidence gait-reference structure?

If yes:
  the student latent is usable, and the previous full policy failed because the
  shared observation backbone / full-action policy bypassed the latent and
  settled into a global trot-biased solution.

If no:
  the student latent is not yet good or stable enough as a control variable,
  even though it is linearly predictive in the offline probe. Then improve the
  history encoder / latent training before more policy-structure work.
```

Design details:

```text
1. Add a diagnostic policy mode for the gait selector:
   selector input = [student latent, command vx]

2. Keep continuous residuals fixed at zero for this first test.

3. Use only high-confidence reference rows:
   selector_aux_min_confidence = 0.25

4. Start with selector reference coefficient 0.15 or 0.20.

5. After training, run:
   - fixed 17-point independent evaluation;
   - information-path probe again;
   - latent replacement test: student latent vs teacher latent vs zero latent
     vs shuffled latent.

Passing criterion:
  gait-selection probabilities must become visibly sensitive to the latent.
  The current mean probability change is only 0.005-0.009; a healthy diagnostic
  should move clearly above this, for example around 0.05 or higher, while also
  recovering ramp 1.0 / ramp 1.5 pronking under high-confidence reference rows.
```

Implementation update:

```text
Implemented the diagnostic mode:
  --selector-latent-cmd-only

When this flag is active:
  - direct task id is forbidden;
  - selector-only mode is required;
  - the gait selector logits are computed only from [command vx, RMA latent];
  - continuous residuals are fixed at zero through selector-only execution;
  - the critic and RMA adaptation still use the full normal observation path.

Updated scripts:
  scripts/train_high_level_ppo.py
  scripts/train_high_level_oracle_ppo.py
  scripts/evaluate_high_level_policy_by_task.py
  scripts/collect_high_level_info_path_data.py

Syntax check passed:
  python3 -m py_compile scripts/train_high_level_ppo.py \
    scripts/train_high_level_oracle_ppo.py \
    scripts/evaluate_high_level_policy_by_task.py \
    scripts/collect_high_level_info_path_data.py
```

First run command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B scripts/train_high_level_oracle_ppo.py \
  --reward-profile canonical_efficiency_v4_physical \
  --allow-diagnostic-reward-profile \
  --no-oracle-condition-obs \
  --selector-only \
  --selector-latent-cmd-only \
  --selector-targets runs/high_level_oracle_gait/selector_targets/20260622_v4_training_range_from_seed208_209/selector_targets.csv \
  --selector-aux-min-confidence 0.25 \
  --selector-aux-coef 0.20 \
  --iterations 100 \
  --num-envs 256 \
  --num-steps 32 \
  --run-name 20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef020_iter100
```

### 2026-06-26 z+speed gait-selector diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef020_iter100
```

Fixed 17-point independent evaluation:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef020_iter100/independent_eval/20260626_training_range_iter099
```

This run removed direct task id, fixed continuous residuals to zero, and forced
the gait selector to use only:

```text
student RMA latent + command vx
```

Result compared with the v4 fair-audit-derived gait reference table:

```text
reference top gait vs learned top gait = 10 / 17
confidence-weighted match = 0.738
high-confidence rows = 6 / 7

learned top gait count:
  pronking = 5 / 17
  trotting = 12 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

High-confidence rows:

```text
flat 1.5:  ref trotting,  learned trotting
flat 2.0:  ref trotting,  learned trotting
ramp 1.0:  ref pronking,  learned pronking
ramp 1.5:  ref pronking,  learned trotting   <-- remaining high-confidence failure
push 1.2:  ref trotting,  learned trotting
push 1.5:  ref trotting,  learned trotting
push 1.8:  ref trotting,  learned trotting
```

Interpretation:

```text
This is a partial success.

Compared with the previous no-task/RMA full-action run, this structure improved
the high-confidence match from 5/7 to 6/7 and confidence-weighted match from
0.637 to 0.738. It recovered ramp 1.0 pronking.

It did not solve the full problem. Ramp 1.5 remains wrong: the reference table
strongly prefers pronking, but the learned selector chooses trotting.

The learned behavior appears to rely strongly on command speed: low-speed rows
tend toward pronking and higher-speed rows tend toward trotting. The next check
must determine whether the selector is actually using the RMA latent more than
before, or whether it is mostly using command speed.
```

Next command should be a new information-path probe on this checkpoint. The key
question is whether replacing student latent with teacher, zero, or shuffled
latent changes gait probabilities much more than the previous 0.005-0.009 range.

### 2026-06-26 z+speed gait-selector information-path result

Analysis:

```text
runs/high_level_oracle_gait/info_path_probe/20260626_notask_zcmd_selector_highconf_coef020/analysis
```

Simple diagnostic accuracy:

```text
history -> task                    test_acc = 0.565
history -> speed                   test_acc = 0.858
history -> reference top gait       test_acc = 0.884

student RMA latent -> task          test_acc = 0.458
student RMA latent -> speed         test_acc = 0.561
student RMA latent -> ref top gait  test_acc = 0.921

teacher latent -> task              test_acc = 0.936
teacher latent -> speed             test_acc = 0.312
teacher latent -> ref top gait      test_acc = 1.000
```

Latent replacement sensitivity:

```text
student latent vs teacher latent:  mean gait-prob diff = 0.0492
student latent vs zero latent:     mean gait-prob diff = 0.0993
student latent vs shuffled latent: mean gait-prob diff = 0.0350
```

Key result:

```text
This is a clear improvement over the previous no-task/RMA full-action policy,
where latent replacement changed gait probabilities by only about 0.005-0.009.

The z+speed selector branch is now using the RMA latent to some degree.
The earlier bottleneck, "the gait selector ignores the latent", is partially
fixed.
```

Remaining failure:

```text
ramp 1.0:
  reference = pronking
  learned top = pronking
  pronking prob = 0.546
  trotting prob = 0.302

ramp 1.5:
  reference = pronking
  learned top = trotting
  pronking prob = 0.361
  trotting prob = 0.417
```

Interpretation:

```text
The remaining problem is no longer total latent non-use. It is a marginal
decision around ramp 1.5. The selector seems to use a strong speed pattern:
lower speed tends pronking, higher speed tends trotting. The RMA latent helps,
but it is not strong enough to override the high-speed trotting tendency for
ramp 1.5.

Do not return to reward editing at this point. v4 physical reward is not the
main bottleneck for this diagnosis.
```

Next minimal plan:

```text
Keep the z+speed selector-only structure.
Run one stronger reference-training diagnostic:
  - no direct task id;
  - continuous residuals fixed;
  - gait selector input = [student RMA latent, command vx];
  - high-confidence reference rows only;
  - stronger reference coefficient or longer training.

Evaluate the same two gates:
  1. fixed 17-point gait match, especially ramp 1.5;
  2. information-path probe, especially latent replacement sensitivity.

If ramp 1.5 remains wrong while latent sensitivity is already nontrivial, the
next target should be improving student latent representation around ramp/rough,
not more reward tuning.
```

### 2026-06-27 stronger z+speed gait-selector result

Run:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100
```

Fixed 17-point independent evaluation:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260627_training_range_iter099
```

This run keeps the same diagnostic structure as the coef 0.20 run:

```text
no direct task id
selector_only = true
selector_latent_cmd_only = true
continuous residuals fixed to zero
gait selector input = [student RMA latent, command vx]
selector_aux_min_confidence = 0.25
```

Only the gait-reference training coefficient changed:

```text
selector_aux_coef: 0.20 -> 0.35
```

Result against the v4 fair-audit-derived gait reference table:

```text
reference top gait vs learned top gait = 11 / 17
confidence-weighted match = 0.879
high-confidence rows = 7 / 7

learned top gait count:
  pronking = 6 / 17
  trotting = 11 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

High-confidence rows:

```text
flat 1.5:  ref trotting,  learned trotting
flat 2.0:  ref trotting,  learned trotting
ramp 1.0:  ref pronking,  learned pronking
ramp 1.5:  ref pronking,  learned pronking
push 1.2:  ref trotting,  learned trotting
push 1.5:  ref trotting,  learned trotting
push 1.8:  ref trotting,  learned trotting
```

Comparison with coef 0.20:

```text
coef 0.20:
  confidence-weighted match = 0.738
  high-confidence rows = 6 / 7
  ramp 1.5 = learned trotting, ref pronking

coef 0.35:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  ramp 1.5 = learned pronking, ref pronking
```

Interpretation:

```text
The stronger high-confidence reference term fixes the remaining high-confidence
ramp 1.5 error. This supports the current diagnosis:

  student RMA latent + command speed contains enough information for the
  high-confidence gait-reference rows;

  the previous coef 0.20 was too weak to overcome the high-speed trotting
  tendency at ramp 1.5.

This is still not a final policy because continuous residuals are fixed to zero.
The result validates the information path and the reference signal under a
controlled diagnostic setting.
```

Next step:

```text
Run the information-path probe on the coef 0.35 checkpoint. We need to verify
that the improved high-confidence gait match still comes with meaningful
RMA-latent sensitivity, rather than simply a stronger speed shortcut.
```

### 2026-06-27 stronger z+speed information-path result

Analysis:

```text
runs/high_level_oracle_gait/info_path_probe/20260627_notask_zcmd_selector_highconf_coef035/analysis
```

Simple diagnostic accuracy:

```text
history -> task                    test_acc = 0.644
history -> speed                   test_acc = 0.856
history -> reference top gait       test_acc = 0.941

student RMA latent -> task          test_acc = 0.576
student RMA latent -> speed         test_acc = 0.608
student RMA latent -> ref top gait  test_acc = 0.950

teacher latent -> task              test_acc = 0.928
teacher latent -> speed             test_acc = 0.338
teacher latent -> ref top gait      test_acc = 1.000
```

Latent replacement sensitivity:

```text
student latent vs teacher latent:  mean gait-prob diff = 0.0521
student latent vs zero latent:     mean gait-prob diff = 0.0770
student latent vs shuffled latent: mean gait-prob diff = 0.0401
```

Important task-speed points:

```text
ramp 1.0:
  reference = pronking
  learned top = pronking
  pronking prob = 0.556
  trotting prob = 0.293

ramp 1.5:
  reference = pronking
  learned top = pronking
  pronking prob = 0.465
  trotting prob = 0.336
```

Interpretation:

```text
The coef 0.35 diagnostic passes the controlled selector-only test.

Compared with coef 0.20, the student latent is more predictive:
  - task: 0.458 -> 0.576
  - speed: 0.561 -> 0.608
  - reference top gait: 0.921 -> 0.950

The gait selector remains meaningfully sensitive to latent replacement:
  0.040-0.077 for shuffled/zero latent,
  0.052 for teacher latent.

This is far stronger than the earlier no-task/RMA full-action baseline where
latent replacement changed gait probabilities by only 0.005-0.009.

Therefore the controlled information path works:
  proprioceptive history -> student RMA latent + command speed -> gait selector.
```

Next stage:

```text
Move from selector-only to full high-level action while preserving the same
information path for gait selection:

  gait selector input = [student RMA latent, command vx]
  continuous residual branch = normal high-level observation/RMA path

This tests whether opening continuous parameters causes the policy to collapse
back to mostly trotting, or whether the stronger gait-reference term and
separate gait-selector branch preserve condition-dependent gait choices.

Implementation detail:
  selector_latent_cmd_only currently requires selector_only. Relax this
  restriction so it can also be used in full-action training, where only the
  gait selector branch is restricted to [student latent, command vx].
```

Implementation update:

```text
scripts/train_high_level_oracle_ppo.py has been updated so
--selector-latent-cmd-only no longer requires --selector-only.

This enables the next diagnostic:
  gait selector branch = [student RMA latent, command vx]
  continuous residual branch = normal high-level policy observation path
  no direct task id
```

Syntax check passed:

```text
python3 -m py_compile scripts/train_high_level_oracle_ppo.py \
  scripts/train_high_level_ppo.py \
  scripts/evaluate_high_level_policy_by_task.py \
  scripts/collect_high_level_info_path_data.py
```

### 2026-06-28 z+speed full-action diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260627_v4_physical_notask_zcmd_fullaction_refprob_highconf_coef035_iter100
```

Fixed 17-point independent evaluation:

```text
runs/high_level_oracle_gait/20260627_v4_physical_notask_zcmd_fullaction_refprob_highconf_coef035_iter100/independent_eval/20260628_training_range_iter099
```

Setup:

```text
no direct task id
selector_latent_cmd_only = true
selector_only = false
continuous residuals enabled
gait selector input = [student RMA latent, command vx]
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Result against the v4 fair-audit-derived gait reference table:

```text
reference top gait vs learned top gait = 9 / 17
confidence-weighted match = 0.723
high-confidence rows = 6 / 7

learned top gait count:
  pronking = 6 / 17
  trotting = 11 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

High-confidence rows:

```text
flat 1.5:  ref trotting,  learned trotting
flat 2.0:  ref trotting,  learned trotting
ramp 1.0:  ref pronking,  learned pronking
ramp 1.5:  ref pronking,  learned trotting   <-- failed after residuals enabled
push 1.2:  ref trotting,  learned trotting
push 1.5:  ref trotting,  learned trotting
push 1.8:  ref trotting,  learned trotting
```

Comparison with selector-only:

```text
selector-only, coef 0.35:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  ramp 1.5 = learned pronking

full-action, coef 0.35:
  confidence-weighted match = 0.723
  high-confidence rows = 6 / 7
  ramp 1.5 = learned trotting
```

Interpretation:

```text
The selector-only information path is valid, but it does not survive naive
full-action training. Once continuous residuals are enabled from scratch, the
policy again finds a path that weakens the gait-selection structure, especially
at ramp 1.5.

This means the next issue is not reward design, not the reference table, and not
complete RMA-latent non-use. The issue is optimization coupling:
continuous-parameter learning can absorb the performance objective and pull the
gait selector away from the high-confidence reference structure.
```

Next plan:

```text
Use staged training:
  1. initialize from the selector-only coef 0.35 checkpoint that passes 7/7
     high-confidence rows;
  2. freeze or strongly protect the gait selector branch;
  3. train continuous residuals for a short diagnostic;
  4. run the same fixed 17-point evaluation;
  5. only if ramp 1.5 pronking survives should we consider carefully unfreezing
     the selector.

This requires adding resume/freeze controls to the high-level training script.
Do not return to reward tuning for this failure.
```

Implementation update:

```text
scripts/train_high_level_oracle_ppo.py now supports staged training controls:

--init-checkpoint
  load an existing high-level checkpoint before training starts

--freeze-latent-cmd-selector
  freeze the [student RMA latent, command vx] gait selector branch

--freeze-rma
  freeze both the student adaptation module and the privileged teacher encoder
```

Syntax check passed:

```text
python3 -m py_compile scripts/train_high_level_oracle_ppo.py \
  scripts/train_high_level_ppo.py \
  scripts/evaluate_high_level_policy_by_task.py \
  scripts/collect_high_level_info_path_data.py
```

Next diagnostic:

```text
Initialize from the selector-only coef 0.35 checkpoint, which passed all 7
high-confidence gait-reference rows. Freeze the gait selector and RMA modules,
then train the full-action policy for a short run. This tests whether continuous
residuals can be learned without pulling ramp 1.5 back to trotting.
```

### 2026-06-28 staged full-action diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_iter100
```

Fixed 17-point independent evaluation:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_iter100/independent_eval/20260628_training_range_iter099
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
no direct task id
selector_latent_cmd_only = true
selector_only = false
continuous residuals enabled
frozen latent_cmd_selector = true
frozen RMA = true
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Result against the v4 fair-audit-derived gait reference table:

```text
reference top gait vs learned top gait = 11 / 17
confidence-weighted match = 0.879
high-confidence rows = 7 / 7

learned top gait count:
  pronking = 6 / 17
  trotting = 11 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

High-confidence rows:

```text
flat 1.5:  ref trotting,  learned trotting
flat 2.0:  ref trotting,  learned trotting
ramp 1.0:  ref pronking,  learned pronking
ramp 1.5:  ref pronking,  learned pronking
push 1.2:  ref trotting,  learned trotting
push 1.5:  ref trotting,  learned trotting
push 1.8:  ref trotting,  learned trotting
```

Comparison:

```text
selector-only coef 0.35:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533

naive full-action coef 0.35:
  confidence-weighted match = 0.723
  high-confidence rows = 6 / 7
  avg reward = 0.7673
  avg vx_err = 0.3499
  ramp 1.5 failed back to trotting

staged full-action from selector-only, frozen selector/RMA:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  avg reward = 0.7681
  avg vx_err = 0.3659
  ramp 1.5 preserved pronking
```

Interpretation:

```text
Staged freezing solves the gait-structure preservation problem. Continuous
residual training no longer destroys ramp 1.5 pronking when the selector and
RMA modules are frozen.

But the continuous residual branch is not yet useful enough: average reward is
lower than selector-only, and average speed error is slightly worse. The learned
continuous parameters shifted, but not in a clearly beneficial direction.
```

Current next question:

```text
How can continuous residuals learn useful parameter adaptation without damaging
the gait selector?
```

Recommended next direction:

```text
Use the staged checkpoint, but reduce the residual learning burden:
  - shorter residual-training run,
  - smaller learning rate,
  - or stronger residual regularization / smaller residual action range.

The immediate next diagnostic should isolate whether the 100-iteration residual
training over-updates the continuous branch. A shorter run from the same
selector-only initialization is the cheapest first test.
```

### 2026-06-28 shorter staged full-action diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_iter030
```

Fixed 17-point independent evaluation:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_iter030/independent_eval/20260628_training_range_iter029
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
no direct task id
selector_latent_cmd_only = true
selector_only = false
continuous residuals enabled
frozen latent_cmd_selector = true
frozen RMA = true
trained for 30 iterations instead of 100
```

Result against the v4 fair-audit-derived gait reference table:

```text
reference top gait vs learned top gait = 11 / 17
confidence-weighted match = 0.879
high-confidence rows = 7 / 7

learned top gait count:
  pronking = 6 / 17
  trotting = 11 / 17
  bounding = 0 / 17
  pacing = 0 / 17
```

Aggregate metrics:

```text
avg reward = 0.7849
avg vx_err = 0.3581
avg gait_switch_rate = 0.0235
avg frequency = 2.9199
avg duration = 0.5041
avg footswing_height = 0.0865
avg stance_width = 0.3314
avg score_clearance = 0.5812
```

Comparison:

```text
selector-only coef 0.35:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg footswing_height = 0.0800
  avg frequency = 3.0000

naive full-action coef 0.35:
  confidence-weighted match = 0.723
  high-confidence rows = 6 / 7
  avg reward = 0.7673
  avg vx_err = 0.3499
  ramp 1.5 failed back to trotting

staged full-action from selector-only, frozen selector/RMA, 100 iterations:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  avg reward = 0.7681
  avg vx_err = 0.3659
  avg footswing_height = 0.0830
  avg frequency = 2.8872

staged full-action from selector-only, frozen selector/RMA, 30 iterations:
  confidence-weighted match = 0.879
  high-confidence rows = 7 / 7
  avg reward = 0.7849
  avg vx_err = 0.3581
  avg footswing_height = 0.0865
  avg frequency = 2.9199
```

Interpretation:

```text
The 30-iteration staged run confirms that freezing the gait selector and RMA
preserves the high-confidence gait structure. The ramp 1.5 pronking failure is
gone, so this is no longer primarily a gait-selection information-path problem.

However, enabling continuous residuals still does not improve performance over
the selector-only checkpoint. The residual branch mainly changes foot swing
height/clearance and frequency, but average reward remains lower and speed error
remains slightly worse than selector-only.

Therefore the current bottleneck is continuous residual adaptation, not reward
definition and not high-confidence gait selection.
```

Current next step:

```text
Do not keep extending the same residual-training run.

Run one low-cost diagnostic that reduces continuous-residual learning pressure:
start from the same selector-only coef 0.35 checkpoint, freeze selector/RMA
again, and train the residual branch with a smaller learning rate. If that only
keeps the policy close to selector-only without improving reward, then add an
explicit residual-magnitude penalty or narrower residual action range before
the next full-action attempt.
```

### 2026-06-29 smaller-learning-rate staged diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_lr1e4_iter030
```

Fixed independent evaluation:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_lr1e4_iter030/independent_eval/20260628_training_range_iter029
```

Additional completed rows:

```text
push_lateral_pace_recovery:1.2
  reward = 0.7528
  vx_err = 0.3610
  pronking ratio = 0.090
  trotting ratio = 0.910

push_lateral_pace_recovery:1.8
  reward = 0.7218
  vx_err = 0.6396
  pronking ratio = 0.029
  trotting ratio = 0.971

stepping_stones_easy_bound_highspeed:1.7
  reward = 0.7364
  vx_err = 0.6291
  pronking ratio = 0.009
  trotting ratio = 0.991
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg gait_switch_rate = 0.0212
  avg footswing_height = 0.0800
  avg frequency = 3.0000
  avg score_clearance = 0.5000

staged full-action, frozen selector/RMA, lr=3e-4, 30 iterations:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.7849
  avg vx_err = 0.3581
  avg gait_switch_rate = 0.0235
  avg footswing_height = 0.0865
  avg frequency = 2.9199
  avg score_clearance = 0.5812

staged full-action, frozen selector/RMA, lr=1e-4, 30 iterations:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.7864
  avg vx_err = 0.3629
  avg gait_switch_rate = 0.0238
  avg footswing_height = 0.0824
  avg frequency = 2.8866
  avg score_clearance = 0.5299
```

Interpretation:

```text
Lowering the learning rate to 1e-4 is not enough. It produces nearly the same
behavior as the 3e-4 staged run. Reward is slightly higher than the 3e-4 staged
run, but speed error is worse, and both full-action staged runs remain worse
than the selector-only baseline.

The continuous residual branch is still not learning useful parameter
adaptation. It changes frequency and foot swing height, but those changes do
not improve the physical objective. The next intervention should be structural:
add a residual-magnitude penalty or narrow the residual action range.
```

Immediate next action:

```text
Implement a residual penalty or residual range scaling before the next
full-action attempt. Do not continue lowering the learning rate or extending the
same residual-training setup.
```

Implementation update:

```text
scripts/evaluate_high_level_policy_by_task.py now has:

--force-zero-residuals

This keeps the model's gait choice, but forces the five continuous residual
actions to zero before environment stepping.
```

New diagnostic columns:

```text
requested_residual_abs_mean
requested_residual_sq_mean
requested_residual_boundary_rate
executed_residual_abs_mean
executed_residual_sq_mean
executed_residual_boundary_rate
```

Purpose:

```text
Run the same full-action checkpoint with learned residuals and with forced-zero
residuals. This directly tests whether the continuous residual output is
responsible for the performance drop relative to selector-only.
```

Next command should evaluate the lr=1e-4 staged checkpoint with
`--force-zero-residuals` on the same 17 training-range rows.

### 2026-06-29 force-zero-residual diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260628_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_lr1e4_iter030/independent_eval/20260629_training_range_iter029_force_zero_residuals
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg frequency = 3.0000
  avg footswing_height = 0.0800

normal staged full-action, lr=1e-4, 30 iterations:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.7864
  avg vx_err = 0.3629
  avg frequency = 2.8866
  avg footswing_height = 0.0824

same checkpoint, forced continuous residuals to zero at evaluation:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8188
  avg vx_err = 0.3561
  avg frequency = 3.0000
  avg footswing_height = 0.0800
  requested_residual_abs_mean = 0.3183
  requested_residual_boundary_rate = 0.0126
  executed_residual_abs_mean = 0.0000
```

Interpretation:

```text
The forced-zero-residual evaluation nearly recovers selector-only reward.
Therefore the performance drop is caused by the continuous residual outputs,
not by gait selection or RMA information use.

The residual output is moderate rather than boundary-saturated. This means the
problem is not mainly action clipping. The branch is learning nonzero offsets
that look plausible in magnitude but reduce the physical objective.
```

Current next step:

```text
Modify staged residual training so the residual branch starts at and stays near
zero unless it earns a clear performance improvement:
  1. zero-initialize the residual output head when loading a selector-only
     checkpoint for full-action staged training;
  2. reduce residual exploration standard deviation for this staged phase;
  3. add a residual-magnitude penalty to the PPO loss.

Do not continue plain full-action staged training without these protections.
```

Implementation update:

```text
scripts/train_high_level_oracle_ppo.py now supports:

--zero-init-residual-head
  reset the residual mean output head to zero after loading an init checkpoint.

--residual-std <value>
  reset the residual exploration standard deviation after checkpoint loading.

--residual-l2-coef <value>
  add a PPO loss penalty on mean squared continuous residual actions.
```

Next experiment:

```text
Start from the selector-only coef 0.35 checkpoint again.
Freeze the gait selector and RMA again.
Enable zero residual-head initialization.
Set residual exploration std lower than the old 0.5.
Add a small residual L2 penalty.
Train 30 iterations.
Evaluate the full 17 training-range rows.
```

### 2026-06-29 protected residual training result

Run:

```text
runs/high_level_oracle_gait/20260629_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_std010_resl2coef003_iter030
```

Fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260629_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_std010_resl2coef003_iter030/independent_eval/20260629_training_range_iter029
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
selector_latent_cmd_only = true
selector_only = false
continuous residuals enabled
frozen latent_cmd_selector = true
frozen RMA = true
zero_init_residual_head = true
residual_std = 0.10
residual_l2_coef = 0.03
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg requested/executed residual abs = 0.0000 / 0.0000
  avg frequency = 3.0000
  avg footswing_height = 0.0800

protected staged full-action:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8193
  avg vx_err = 0.3532
  avg requested residual abs = 0.0418
  avg executed residual abs = 0.0257
  avg requested residual boundary rate = 0.0000
  avg frequency = 3.0010
  avg footswing_height = 0.0807

previous plain staged full-action, 30 iterations:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.7849
  avg vx_err = 0.3581
  avg frequency = 2.9199
  avg footswing_height = 0.0865

previous low-lr staged full-action, forced residuals to zero:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8188
  avg vx_err = 0.3561
  avg executed residual abs = 0.0000
```

Interpretation:

```text
The residual protections worked as a safety mechanism. They prevent the
continuous branch from damaging the already-correct gait selector and keep the
policy near the selector-only/default-template performance level.

However, the protected continuous branch does not yet provide meaningful
additional adaptation. The fixed evaluation is essentially tied with the
selector-only checkpoint: reward is slightly lower, speed error is essentially
unchanged, and the learned residuals are very small.

Per-row deltas versus selector-only show small mixed effects. The protected
residuals slightly improve rough-slope rows and some high-speed rows, but hurt
several push/stones rows and do not create a consistent tracking improvement.
They mostly add a small clearance increase rather than a clear task-conditioned
parameter adaptation.
```

Current conclusion:

```text
The immediate residual failure mode has been fixed: continuous parameters no
longer destroy performance when zero-initialized, low-variance, and L2
regularized.

The remaining problem is usefulness, not safety. The continuous residual branch
has learned to stay near the default WTW gait templates, but it has not learned
a reliable improvement over them.
```

Next decision:

```text
Do not go back to reward editing or gait-selector debugging.

The next residual diagnostic should either:
  1. keep the current safe setup but relax it slightly, for example residual_std
     0.15 or residual_l2_coef 0.01, to see whether small-but-nonzero
     adaptations can improve reward without recreating the old damage; or
  2. introduce a more structured residual curriculum, first allowing only one
     parameter family such as frequency or footswing height, then adding others
     only if fixed evaluation improves.

The pass criterion is not larger residual magnitude. The pass criterion is a
real fixed-evaluation gain over selector-only while preserving the 7 / 7
high-confidence gait-reference rows.
```

### 2026-06-30 relaxed protected residual result

Run:

```text
runs/high_level_oracle_gait/20260629_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_std015_resl2coef001_iter030
```

Fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260629_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_std015_resl2coef001_iter030/independent_eval/20260629_training_range_iter029
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
selector_latent_cmd_only = true
selector_only = false
continuous residuals enabled
frozen latent_cmd_selector = true
frozen RMA = true
zero_init_residual_head = true
residual_std = 0.15
residual_l2_coef = 0.01
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg executed residual abs = 0.0000
  avg frequency = 3.0000
  avg footswing_height = 0.0800

protected residual, std=0.10, residual_l2_coef=0.03:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8193
  avg vx_err = 0.3532
  avg requested residual abs = 0.0418
  avg executed residual abs = 0.0257
  avg frequency = 3.0010
  avg footswing_height = 0.0807

relaxed protected residual, std=0.15, residual_l2_coef=0.01:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8178
  avg vx_err = 0.3532
  avg requested residual abs = 0.0632
  avg executed residual abs = 0.0341
  avg requested residual boundary rate = 0.0000
  avg frequency = 3.0140
  avg footswing_height = 0.0798
```

Interpretation:

```text
Relaxing all continuous residual dimensions together is not useful. It preserves
the gait-reference structure, so the gait selector remains healthy, but it does
not improve the physical objective. Average reward drops below both the
selector-only checkpoint and the more conservative protected residual run.

The learned residuals are larger than the conservative run, but still not
boundary-saturated. Therefore the problem is not that the residual branch is too
constrained by clipping. The problem is that unconstrained multi-parameter
residual changes do not produce a coherent improvement over the WTW default
templates.

Per-row deltas versus selector-only are mixed:
  - rough-slope speed tracking improves on several rows,
  - push 1.2 / 1.5 and some flat/ramp rows get worse,
  - clearance drops below the conservative protected run,
  - there is no reliable average reward gain.
```

Current conclusion:

```text
The "make all residuals a bit freer" route should stop here. It only increases
residual magnitude without producing useful adaptation.

The next residual work should be structured rather than globally freer:
  1. open one continuous parameter family at a time;
  2. start with the most interpretable candidates, frequency and footswing
     height;
  3. keep the same safe setup: start from selector-only, freeze gait selector
     and RMA, zero-init residual head, small residual std, residual L2 penalty;
  4. compare each one-parameter diagnostic against selector-only and the
     std=0.10 protected run on the same fixed 17 rows.

Only if one parameter family gives a real fixed-evaluation gain should it be
combined with another parameter family.
```

Implementation update:

```text
scripts/train_high_level_ppo.py now supports a non-persistent residual action
mask inside ActorCritic. Masked residual dimensions are held at zero and do not
contribute to residual log-probability or entropy.

scripts/train_high_level_oracle_ppo.py now supports:

--residual-train-dims <dims>

Examples:
  --residual-train-dims frequency
  --residual-train-dims footswing_height
  --residual-train-dims frequency,footswing_height
  --residual-train-dims all
  --residual-train-dims none

scripts/evaluate_high_level_policy_by_task.py and
scripts/collect_high_level_info_path_data.py read the residual mask from the
run's args.json, so fixed evaluation and information-path collection execute
the same residual dimensions that were trained.
```

Immediate next diagnostic:

```text
Run a one-parameter residual diagnostic starting with frequency only.

Reason:
  frequency is the cleanest first test because it directly affects cadence and
  speed tracking. The relaxed all-parameter run showed small mixed tracking
  changes but no reward gain. Testing frequency alone tells us whether cadence
  adaptation has value without confounding it with foot height, stance width,
  duration, and body pitch.

Pass criterion:
  fixed 17-row avg reward > selector-only 0.8195, or at least clear speed-error
  improvement without losing 7 / 7 high-confidence gait-reference rows.
```

Refined training interpretation:

```text
The issue should not be described as "continuous residuals make the gradient
wrong." A more accurate interpretation is:

Early random continuous-parameter exploration makes reward attribution noisy.
The reward changes can come from gait choice, from perturbed frequency/height/
width/pitch parameters, or from their interaction. This makes the gait selector
harder to train cleanly.

Therefore the active training strategy is staged:
  1. first train the gait selector with zero continuous residuals;
  2. after high-confidence gait rows are stable, freeze/protect the selector;
  3. open one continuous parameter at a time from zero;
  4. keep only parameter families that beat selector-only on the fixed 17-row
     evaluation while preserving the 7 / 7 high-confidence gait rows.

This does not assume continuous parameters are useless. It tests whether each
parameter family has measurable value after gait selection is already stable.
```

### 2026-06-30 frequency-only residual diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260630_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_frequency_only_std015_resl2coef001_iter030
```

Fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260630_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_frequency_only_std015_resl2coef001_iter030/independent_eval/20260630_training_range_iter029
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
selector_latent_cmd_only = true
selector_only = false
frozen latent_cmd_selector = true
frozen RMA = true
zero_init_residual_head = true
residual_train_dims = frequency
residual_std = 0.15
residual_l2_coef = 0.01
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg frequency = 3.0000

frequency-only residual:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8187
  avg vx_err = 0.3531
  avg requested residual abs = 0.0154
  avg executed residual abs = 0.0084
  avg frequency = 2.9924

protected all-residual, std=0.10, residual_l2_coef=0.03:
  avg reward = 0.8193
  avg vx_err = 0.3532
  avg executed residual abs = 0.0257

relaxed all-residual, std=0.15, residual_l2_coef=0.01:
  avg reward = 0.8178
  avg vx_err = 0.3532
  avg executed residual abs = 0.0341
```

Interpretation:

```text
Frequency-only tuning preserves the high-confidence gait structure, but it does
not provide a meaningful fixed-evaluation gain over selector-only/default
parameters.

The tiny vx_err improvement is too small to treat as a real success, especially
because average reward is lower than selector-only and lower than the
conservative protected all-residual run.

Per-row deltas are mixed:
  - rough-slope 0.5 / 1.0 improve modestly;
  - push 1.2 / 1.5 get worse;
  - high-speed push and some stones rows improve, but not enough to lift the
    aggregate score.

Therefore cadence/frequency alone is not a reliable residual adaptation under
the current setup.
```

Next residual diagnostic:

```text
Test footswing_height only.

Reason:
  footswing height is the next most interpretable single parameter. It affects
  clearance and contact safety without directly changing cadence. Previous
  all-parameter runs often changed foot height/clearance, but those runs were
  confounded with frequency, stance width, duration, and body pitch. Testing
  footswing_height alone isolates whether clearance adaptation has independent
  value.

If footswing_height-only also fails to beat selector-only, stop single-parameter
testing for now and treat the current default WTW continuous parameters as a
strong baseline. Further work would then need a better continuous-parameter
objective or a different parameterization, not more blind residual training.
```

### 2026-06-30 footswing-height-only residual diagnostic result

Run:

```text
runs/high_level_oracle_gait/20260630_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_footswing_only_std015_resl2coef001_iter030
```

Fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260630_v4_physical_notask_zcmd_fullaction_from_selector_coef035_freeze_selector_rma_zerohead_footswing_only_std015_resl2coef001_iter030/independent_eval/20260630_training_range_iter029
```

Setup:

```text
initialized from selector-only coef 0.35 checkpoint
selector_latent_cmd_only = true
selector_only = false
frozen latent_cmd_selector = true
frozen RMA = true
zero_init_residual_head = true
residual_train_dims = footswing_height
residual_std = 0.15
residual_l2_coef = 0.01
selector_aux_min_confidence = 0.25
selector_aux_coef = 0.35
```

Full 17-row comparison:

```text
selector-only coef 0.35:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8195
  avg vx_err = 0.3533
  avg footswing_height = 0.0800
  avg clearance score = 0.5000

footswing-height-only residual:
  match = 11 / 17
  high-confidence rows = 7 / 7
  avg reward = 0.8191
  avg vx_err = 0.3513
  avg requested residual abs = 0.0182
  avg executed residual abs = 0.0089
  avg footswing_height = 0.0804
  avg clearance score = 0.5050

frequency-only residual:
  avg reward = 0.8187
  avg vx_err = 0.3531
  avg executed residual abs = 0.0084
```

Interpretation:

```text
Footswing-height-only tuning preserves the high-confidence gait structure and
does what it is mechanically expected to do: it slightly raises clearance and
slightly lowers the average speed error.

However, it does not beat the selector-only/default-parameter baseline on the
main fixed-evaluation reward. The average reward is lower than selector-only
0.8195, and the per-row effect is mixed:
  - rough rows and stones rows improve modestly;
  - push 1.5 becomes clearly worse;
  - flat/ramp changes are small and not consistently positive.

Therefore this is not a clean pass. Footswing-height adaptation may contain a
small useful signal, but under the current objective and parameterization it is
not robust enough to justify opening it by default.
```

Current decision:

```text
Frequency-only and footswing-height-only diagnostics both preserve the gait
selector but fail to produce a robust gain over the zero-residual/default-WTW
continuous parameters.

Do not continue blindly through the remaining residual dimensions just because
they exist. The active baseline should remain:
  gait selector trained first;
  continuous residuals fixed at zero;
  continuous parameters opened only when there is a specific, testable physical
  reason and a clear fixed-evaluation pass criterion.

If continuous tuning is revisited, it should use a better parameterization or a
more targeted objective, not more unconstrained or weakly constrained residual
training.
```

### 2026-07-02 current reference-supervised diagnostic baseline summary

The active diagnostic baseline is now the selector-only/default-continuous-
parameter policy:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100
```

Fixed 17-row evaluation:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260627_training_range_iter099
```

Human-readable and CSV comparison tables:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260627_training_range_iter099/main_baseline_reference_comparison.md
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260627_training_range_iter099/main_baseline_reference_comparison.csv
```

Setup:

```text
direct task id = false
gait selector input = student RMA latent + command vx
continuous residuals = fixed at zero/default WTW gait-template parameters
selector reference target = 20260622_v4_training_range_from_seed208_209
```

Aggregate fixed-evaluation result:

```text
rows = 17
reference top-gait match = 11 / 17
high-confidence reference rows = 7 / 7
confidence-weighted match = 0.879
avg reward = 0.8195
avg vx_err = 0.3533
avg done_rate = 0.0209
avg gait_switch_rate = 0.0212
learned top-gait counts = pronking 6, trotting 11
```

Interpretation:

```text
This is currently the cleanest reference-supervised diagnostic baseline:
  - it does not receive direct task labels;
  - it uses proprioceptive/RMA evidence plus command speed;
  - it preserves all high-confidence gait-reference rows;
  - it avoids the early continuous-parameter exploration problem;
  - it beats the tested one-parameter residual variants on average reward.

The low-confidence mismatches should not be over-interpreted because the
reference table itself has small margins there. The policy expresses pronking
and trotting as dominant gait families; bounding and pacing do not appear as top
gaits under the current reference/training setup.
```

Current technical route:

```text
Stop treating continuous residuals as the immediate next lever.

Use this selector-only/default-parameter policy as the reference-supervised
diagnostic baseline, then test whether it is genuinely condition-aware:
  1. interpolation-speed evaluation inside the training speed ranges;
  2. same-speed cross-condition evaluation to check whether terrain/condition
     changes alter gait probabilities beyond simple speed thresholding;
  3. information-path checks only if the policy still appears to rely mostly on
     speed rather than condition evidence.
```

### 2026-07-02 interpolation-speed evaluation

Run:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260702_interpolation_iter099
```

Purpose:

```text
Check whether the current reference-supervised diagnostic baseline only works
at the fixed 17 training speed points, or whether it behaves smoothly at
intermediate command speeds.
```

Setup:

```text
direct task id = false
gait selector input = student RMA latent + command vx
continuous residuals = fixed at zero/default WTW parameters
num_envs = 32
steps = 1000
evaluated rows = 12
```

Aggregate:

```text
avg reward = 0.8240
avg vx_err = 0.3206
avg done_rate = 0.0211
avg gait_switch_rate = 0.0260
```

Per-condition result:

```text
flat:
  vx 0.75 -> trotting 0.533, pronking 0.467
  vx 1.25 -> trotting 0.999
  vx 1.75 -> trotting 1.000

ramp:
  vx 0.75 -> pronking 0.997
  vx 1.25 -> pronking 0.854
  vx 1.75 -> pronking 0.543, trotting 0.457

rough:
  vx 0.75 -> pronking 0.889
  vx 1.25 -> pronking 0.526, trotting 0.474
  vx 1.75 -> trotting 0.646

push:
  vx 1.35 -> trotting 0.956
  vx 1.65 -> trotting 0.971

stepping stones:
  vx 1.85 -> trotting 0.994
```

Interpretation:

```text
This is a positive sanity check for the reference-supervised diagnostic
baseline. It does not appear to memorize only the fixed 17 evaluation speeds.
The intermediate-speed behavior is smooth and consistent with neighboring
fixed-speed rows.

The most important transitions are plausible:
  - flat moves from low-speed pronking/trotting mixture to stable trotting;
  - ramp stays pronking-dominant through the high-confidence mid-speed region;
  - rough gradually shifts from pronking to trotting as speed increases;
  - push stays trotting.

However, this does not prove full condition awareness. The policy still uses
only pronking and trotting as dominant gait families, and part of the behavior
may still be explained by speed thresholds. The next test should compare
different conditions at the same command speed.
```

Same-speed condition check from the combined fixed-speed and interpolation
evaluations:

```text
vx 1.00:
  flat  -> pronking 0.017, trotting 0.983
  ramp  -> pronking 0.975, trotting 0.025
  rough -> pronking 0.680, trotting 0.320

vx 1.50:
  flat  -> pronking 0.000, trotting 1.000
  ramp  -> pronking 0.731, trotting 0.269
  rough -> pronking 0.417, trotting 0.583

vx 1.75:
  flat  -> pronking 0.000, trotting 1.000
  ramp  -> pronking 0.543, trotting 0.457
  rough -> pronking 0.354, trotting 0.646
```

Updated interpretation:

```text
The current reference-supervised diagnostic baseline is not a pure command-speed
threshold policy. At the same command speed, flat / ramp / rough produce clearly
different gait probabilities:
  - flat is strongly trotting-biased at medium/high speed;
  - ramp remains much more pronking-biased;
  - rough usually sits between flat and ramp.

This is positive evidence that the gait selector uses condition information in
addition to command speed.
```

Remaining limitation:

```text
The policy still expresses only pronking and trotting as top gait families.
Bounding and pacing are not active as dominant gaits under the current
reference-supervised diagnostic baseline.

Next work should therefore shift from proving basic condition awareness to
testing robustness/generalization of this diagnostic baseline:
  1. run a small new-seed repeat of the diagnostic baseline evaluation if
     variance is a concern;
  2. test out-of-range or harder scenarios only after deciding what real-world
     capability is being claimed;
  3. keep continuous parameters fixed unless a new parameterization has a clear
     physical hypothesis.
```

### 2026-07-04 fixed 17-row repeat evaluation

Repeat evaluation artifacts:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260703_training_range_repeat_iter099
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260703_training_range_repeat_missing3_iter099
```

Combined 17-row repeat:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/independent_eval/20260703_training_range_repeat_combined17_iter099
```

Important correction:

```text
The first repeat command used --full, but --full was still the old 14-row list.
It omitted:
  push_lateral_pace_recovery:1.2
  push_lateral_pace_recovery:1.8
  stepping_stones_easy_bound_highspeed:1.7

The missing three rows were run separately and then combined with the 14-row
repeat to form the complete 17-row repeat.

evaluate_high_level_policy_by_task.py has been updated so --full now covers the
current 17-row training-range set.
```

Aggregate comparison:

```text
original fixed 17-row evaluation:
  avg reward = 0.8195
  avg vx_err = 0.3533
  done_rate = 0.0209
  gait_switch_rate = 0.0212
  reference top-gait match = 11 / 17
  high-confidence match = 7 / 7
  confidence-weighted match = 0.879
  top-gait counts = pronking 6, trotting 11

repeat fixed 17-row evaluation:
  avg reward = 0.8189
  avg vx_err = 0.3546
  done_rate = 0.0209
  gait_switch_rate = 0.0215
  reference top-gait match = 11 / 17
  high-confidence match = 7 / 7
  confidence-weighted match = 0.879
  top-gait counts = pronking 6, trotting 11
```

Interpretation:

```text
The current reference-supervised diagnostic baseline is repeat-stable on the
fixed 17-row evaluation.

All top-gait decisions are unchanged from the original evaluation. The average
reward and speed-error differences are tiny:
  reward: 0.8195 -> 0.8189
  vx_err: 0.3533 -> 0.3546

The only notable row-level drift is push 1.2 / push 1.5 speed error becoming a
little worse, but this does not change the gait decision or the high-confidence
reference match.

This strengthens the decision to use the selector-only/default-continuous policy
as the active reference-supervised diagnostic baseline.
```

Updated next step:

```text
Do not continue repeating the same fixed 17-row evaluation.

The next useful work for this diagnostic baseline is robustness/generalization:
  - either define a harder or out-of-range evaluation set;
  - or package the current diagnostic baseline as the reference policy for
    visual and route-style tests.

Continuous residual tuning remains paused.
```

### 2026-07-04 single-scene visual checks

Policy:

```text
runs/high_level_oracle_gait/20260626_v4_physical_notask_zcmd_selector_refprob_highconf_coef035_iter100/checkpoints/high_level_000099.pt
```

Visual checks were run one scene at a time because remote-desktop mouse
navigation made the multi-scene viewer difficult to inspect.

Observed behavior:

```text
ramp_up_trot_robustness vx=1.0:
  Mostly pronking, matching the expected high-confidence behavior.
  Visible issue: yaw/heading drift or horizontal drift can accumulate when the
  body direction becomes slightly biased.

rough_slope_trot_robustness vx=1.0:
  Pronking and trotting alternate.
  This is acceptable because rough 1.0 is a boundary case numerically rather
  than a high-confidence single-gait row.

flat_trot_efficiency vx=1.0:
  Almost entirely trotting.
  One occasional initial pronking episode appears related to reset/landing
  state and does not persist.

push_lateral_pace_recovery vx=1.5:
  Mostly trotting.
  After a lateral push, the policy can briefly use pronking for about one
  second, then returns to trotting. This looks like a plausible recovery
  response rather than a failure.

stepping_stones_easy_bound_highspeed vx=2.0:
  Almost entirely trotting.
  Tripping, speed loss, and falls occur, which is expected given the depth of
  the gaps in the current scene. The baseline does not express bounding or
  pacing as a gap-crossing strategy.
```

Interpretation:

```text
The visual checks support the numerical conclusion:
  - flat/ramp/rough/push show meaningful pronking/trotting condition response;
  - push includes a potentially useful short pronking recovery after perturbation;
  - stones remains a clear limitation.

The stepping-stones result should not be hidden or interpreted as successful
gap-crossing. Under the current reference-supervised diagnostic baseline, stones
is handled mostly by trotting and is not robust to the deeper gaps in the
visualization scene.

This limitation does not invalidate the diagnostic conclusion that the
architecture can express pronking/trotting condition response when trained with
a gait-reference signal. It means that if stones/gap-crossing is a required
capability, it should become a separate capability target with dedicated terrain
design and possibly stronger low-level or template support for bounding/pacing.
```

Next decision:

```text
The current reference-supervised diagnostic baseline can now be summarized as a
stable pronking/trotting condition-aware policy with default continuous
parameters.

Do not claim it solves all gait families or deep stepping-stones traversal.
The next project step should be chosen based on the desired claim:
  A. package the current baseline as a diagnostic / upper-bound result and
     document limitations;
  B. start a separate stones/gap-crossing capability line;
  C. return to the clean no-reference training line if the goal is the strongest
     generalization claim;
  D. build a route-style test after updating the route script to the current
     no-task/RMA observation contract.
```

### 2026-07-09 decision-interval ramp/flat diagnosis

Goal:

```text
Test whether long gait execution windows can solve the earlier problem where
short-horizon training collapsed to trotting and could not discover stable ramp
pronking.
```

Key change:

```text
Use decision_interval=50 with selector_hold_steps=0.
One high-level gait choice is executed for 50 high-level environment steps, and
PPO receives one transition for that whole decision window.
```

Reason:

```text
Paired same-state audits showed that pronking can look worse than trotting over
short horizons when it starts from a trotting-context state, but becomes better
over longer horizons on ramp. Therefore short-horizon training can assign bad
credit to pronking even when sustained pronking is better.
```

Ramp-only result:

```text
Run:
  runs/high_level_oracle_gait/20260709_v4_ramp_only_decision_interval50_phys800_iter050

Training last 10 iterations:
  reward = 0.8312
  vx_err = 0.1937
  pronking ratio = 0.873
  trotting ratio = 0.126

Fixed-speed ramp evaluation:
  vx 0.5 / 1.0 / 1.5 / 2.0 all became pronking.
```

Interpretation:

```text
The long decision window can indeed make ramp learn pronking without using task
labels or gait-reference supervision. This confirms that the earlier all-trot
result was at least partly a temporal credit-assignment problem.
```

Ramp-to-flat continuation result:

```text
Run:
  runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_from_ramp_iter050

Result:
  both flat and ramp became essentially all pronking.
```

Flat fixed-gait check:

```text
Run:
  runs/high_level_oracle_gait/fixed_gait_live_reward_audit/20260709_v4_flat_default_trot_pronk_check

Fixed trotting beats fixed pronking on flat at every checked speed:
  vx 0.5: +0.0013
  vx 1.0: +0.0068
  vx 1.5: +0.0189
  vx 2.0: +0.0407

Average:
  fixed trotting = 0.8340
  fixed pronking = 0.8171
```

Interpretation:

```text
The flat all-pronking behavior is not the true fixed-gait optimum under the v4
physical reward. It is a training outcome / optimization attractor.
```

Ramp+flat from-scratch result:

```text
Run:
  runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_scratch_iter050

Training:
  first 5 iterations:
    reward = 0.8109
    vx_err = 0.2048
    pronking = 0.253
    trotting = 0.244

  iterations 20-29:
    reward = 0.8368
    vx_err = 0.1893
    pronking = 0.925
    trotting = 0.057

  last 10 iterations:
    reward = 0.8373
    vx_err = 0.1899
    pronking = 0.989
    trotting = 0.008
```

Fixed-speed evaluation:

```text
Run:
  runs/high_level_oracle_gait/20260709_v4_ramp_flat_decision_interval50_phys800_scratch_iter050/independent_eval/20260709_flat_ramp_iter049

Flat:
  vx 0.5: pronking 1.000, reward 0.9154, vx_err 0.0592
  vx 1.0: pronking 1.000, reward 0.8663, vx_err 0.1355
  vx 1.5: pronking 1.000, reward 0.7929, vx_err 0.2711
  vx 2.0: pronking 1.000, reward 0.6966, vx_err 0.4831
  average reward = 0.8178

Ramp:
  vx 0.5: pronking 1.000, reward 0.9031, vx_err 0.0773
  vx 1.0: pronking 1.000, reward 0.8542, vx_err 0.1447
  vx 1.5: pronking 1.000, reward 0.7803, vx_err 0.2851
  vx 2.0: pronking 0.9999, reward 0.6765, vx_err 0.5205
  average reward = 0.8035
```

Updated interpretation:

```text
Long decision windows solve one problem but expose another:

  short horizon:
    all-trotting attractor; ramp pronking is not discovered.

  decision_interval=50:
    ramp pronking is discovered, but pronking becomes a global attractor and
    spreads to flat even though fixed-gait evaluation says flat should prefer
    trotting.

This means the next issue is not simply terrain observability and not simply the
reward definition. The training update itself is not preserving the flat
trotting advantage under mixed flat+ramp training.
```

Next diagnostic:

```text
Do not add rough/push/stones yet.

First isolate whether the all-pronking outcome comes from:
  1. decision_interval=50 itself making flat prefer pronking during PPO;
  2. ramp examples dominating the shared policy update and contaminating flat;
  3. the advantage signal in the mixed rollout assigning positive credit to
     flat pronking despite fixed-gait flat trotting being better.

The smallest next run is flat-only with decision_interval=50 from scratch.
If flat-only learns trotting, then ramp is contaminating flat in mixed training.
If flat-only also learns pronking, then the long-window PPO setup itself has a
pronking bias even on flat.
```

Flat-only diagnostic result:

```text
Run:
  runs/high_level_oracle_gait/20260710_v4_flat_only_decision_interval50_phys800_scratch_iter050

Training:
  first 5 iterations:
    reward = 0.8297
    vx_err = 0.1835
    pronking = 0.396
    trotting = 0.253
    gait_switch_rate = 0.708

  iterations 20-29:
    reward = 0.8441
    vx_err = 0.1822
    pronking = 0.972
    trotting = 0.018
    gait_switch_rate = 0.054

  last 10 iterations:
    reward = 0.8444
    vx_err = 0.1823
    pronking = 0.999
    trotting = 0.001
    gait_switch_rate = 0.003

  final iteration:
    pronking = 1.000
    trotting = 0.000
```

Updated interpretation:

```text
The all-pronking result is not caused only by ramp examples contaminating flat.
With decision_interval=50, even flat-only training converges to pronking.

This is especially important because the fixed-gait flat audit showed that
fixed trotting has higher reward than fixed pronking on flat. Therefore the
problem is not simply the mixed terrain distribution. The long-window PPO
training setup is producing a pronking attractor on flat despite fixed-gait
reward comparisons favoring trotting.
```

Next diagnostic:

```text
Run fixed-speed evaluation for the flat-only checkpoint. If the fixed-speed eval
also shows all-pronking across 0.5 / 1.0 / 1.5 / 2.0, then the training result is
stable and not just a rollout-statistics artifact.

After that, inspect per-gait advantage / reward statistics under flat-only
decision_interval=50 to answer why PPO reinforces pronking even when fixed-gait
flat audit favors trotting.
```

Flat-only fixed-speed evaluation:

```text
Run:
  runs/high_level_oracle_gait/20260710_v4_flat_only_decision_interval50_phys800_scratch_iter050/independent_eval/20260710_flat_full_iter049

Result:
  vx 0.5: pronking 1.000, reward 0.9152, vx_err 0.0601
  vx 1.0: pronking 1.000, reward 0.8655, vx_err 0.1362
  vx 1.5: pronking 1.000, reward 0.7925, vx_err 0.2745
  vx 2.0: pronking 1.000, reward 0.6946, vx_err 0.4892

Average:
  reward = 0.8170
  vx_err = 0.2400
  pronking = 1.000
  trotting = 0.000
```

Comparison with fixed-gait flat audit:

```text
The learned flat-only policy is effectively the fixed-pronking policy:
  learned flat-only avg reward = 0.8170
  fixed pronking avg reward = 0.8171
  fixed trotting avg reward = 0.8340

Therefore the training procedure is selecting a suboptimal flat gait under the
same v4 physical reward.
```

Preliminary per-gait training signal:

```text
Flat-only decision_interval=50, sampled-gait statistics:

First 5 iterations:
  pronking: count 2028, adv +0.0055, reward 20.263
  trotting: count 1294, adv +0.0536, reward 20.555

Iterations 5-9:
  pronking: count 3029, adv +0.0350, reward 20.583
  trotting: count 773,  adv +0.0805, reward 20.884

Iterations 20-29:
  pronking: count 9957, adv +0.0042, reward 20.855
  trotting: count 182,  adv -0.0540, reward 20.454

Last 10 iterations:
  pronking: count 10227, adv +0.0007, reward 20.928
  trotting: count 12,    adv -0.5906, reward 21.462
```

Interpretation:

```text
Early in training, trotting was not worse in the rollout statistics; it had
higher sampled reward and higher sampled advantage than pronking. Despite that,
the policy moved toward pronking and then stopped sampling trotting enough for
it to recover.

This makes the problem more specific:
  - not mixed-terrain contamination;
  - not fixed-gait reward preference for pronking;
  - likely a training-update / action-credit / categorical-gait mapping issue
    introduced or exposed by decision_interval=50.
```

Next implementation check:

```text
Inspect the decision_interval=50 PPO data path:
  - which gait index is stored as the action for the 50-step decision;
  - whether log probability uses the same gait index as the executed gait;
  - whether the gait metric names match the categorical action order;
  - whether accumulated 50-step reward is assigned to the correct decision;
  - whether advantage normalization or masking could invert the useful signal.

Do this before adding any more terrain or changing the reward.
```

Code inspection update:

```text
The sampled gait and executed gait are aligned in the flat-only run:
  sampled_executed_gait_mismatch_rate = 0.0

The PPO path stores the sampled one-hot gait action, re-evaluates log probability
from the same one-hot action, and accumulates the 50-step reward into the same
decision transition. No obvious gait-index mismatch was found in the inspected
path.
```

New diagnostic option:

```text
Added:
  --zero-init-selector-head

Meaning:
  reset the gait selector's final linear layer to zero, so the initial gait
  logits are exactly uniform across all gait families.

Reason:
  In the flat-only decision_interval=50 run, early sampling was already biased
  toward pronking:
    first 5 iterations: pronking 0.396, trotting 0.253

  This happened even though early sampled rollout statistics did not show
  trotting to be worse than pronking. With only 16 high-level decisions per
  environment per iteration, an initial categorical bias can be amplified into a
  stable gait collapse.

  Zero-initializing the selector head does not inject a trotting prior. It only
  removes random initial gait preference so the next flat-only test can check
  whether pronking still wins from an unbiased start.
```
