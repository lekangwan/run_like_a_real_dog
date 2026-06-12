# Oracle Training Metrics Summary

- run_dir: `runs/high_level_oracle_gait/20260610_rma_notask_reward_v4`
- rows: 100
- iterations: 0 to 99

## Early vs Late

| metric | early | late | delta |
|---|---:|---:|---:|
| reward | 0.716585 | 0.702279 | -0.014306 |
| weighted_metric_reward | 0.716585 | 0.702279 | -0.014306 |
| vx_err | 0.418400 | 0.439753 | 0.021352 |
| lateral_position_penalty | 3.196788 | 3.315368 | 0.118580 |
| gait_switch_penalty | 0.118176 | 0.117688 | -0.000488 |
| gait_switch_rate | 0.234350 | 0.235572 | 0.001222 |
| score_progress | 0.627076 | 0.606262 | -0.020814 |
| score_clearance | 0.735539 | 0.777257 | 0.041718 |
| score_gait_stability | 0.795634 | 0.796479 | 0.000844 |
| score_action_smoothness | 0.418629 | 0.357223 | -0.061406 |
| score_action_magnitude | 0.765678 | 0.691398 | -0.074280 |
| score_action_boundary_margin | 0.999997 | 0.999764 | -0.000233 |
| action_clip_rate | 0.000000 | 0.000002 | 0.000002 |
| footswing_height_mean | 0.098823 | 0.102717 | 0.003893 |
| stance_width_mean | 0.352335 | 0.350107 | -0.002228 |
| body_pitch_mean | 0.002013 | 0.002059 | 0.000045 |

## Per-Task Action Health

| task | clip early | clip late | switch early | switch late | footswing early | footswing late |
|---|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 0.000000 | 0.000000 | 0.234367 | 0.235422 | 0.098982 | 0.101708 |
| ramp_up_trot_robustness | 0.000000 | 0.000012 | 0.234535 | 0.236369 | 0.098865 | 0.102281 |
| rough_slope_trot_robustness | 0.000000 | 0.000000 | 0.232701 | 0.235294 | 0.098652 | 0.102222 |
| push_lateral_pace_recovery | 0.000000 | 0.000000 | 0.236116 | 0.234978 | 0.098810 | 0.102046 |
| stepping_stones_easy_bound_highspeed | 0.000000 | 0.000000 | 0.234029 | 0.235800 | 0.098904 | 0.102655 |

## Target Gait Ratios

| task | target ratio early | target ratio late | delta |
|---|---:|---:|---:|
| flat_trot_efficiency | 0.235757 | 0.336599 | 0.100841 |
| ramp_up_trot_robustness | 0.234804 | 0.342279 | 0.107475 |
| rough_slope_trot_robustness | 0.234130 | 0.344547 | 0.110417 |
| push_lateral_pace_recovery | 0.279718 | 0.266912 | -0.012806 |
| stepping_stones_easy_bound_highspeed | 0.193873 | 0.255515 | 0.061642 |

## Baseline Late vs Current Late

| metric | baseline late | current late | delta |
|---|---:|---:|---:|
| reward | 0.661354 | 0.702279 | 0.040925 |
| weighted_metric_reward | 0.661354 | 0.702279 | 0.040925 |
| vx_err | 0.454113 | 0.439753 | -0.014360 |
| lateral_position_penalty | 3.412257 | 3.315368 | -0.096888 |
| gait_switch_penalty | 0.117438 | 0.117688 | 0.000250 |
| gait_switch_rate | 0.235496 | 0.235572 | 0.000076 |
| score_progress | 0.600482 | 0.606262 | 0.005780 |
| score_clearance | 0.707930 | 0.777257 | 0.069327 |
| score_gait_stability | 0.796911 | 0.796479 | -0.000433 |
| score_action_smoothness | 0.358554 | 0.357223 | -0.001330 |
| score_action_magnitude | 0.694945 | 0.691398 | -0.003547 |
| score_action_boundary_margin | 0.999845 | 0.999764 | -0.000081 |
| action_clip_rate | 0.000000 | 0.000002 | 0.000002 |
| footswing_height_mean | 0.097148 | 0.102717 | 0.005568 |
| stance_width_mean | 0.352695 | 0.350107 | -0.002587 |
| body_pitch_mean | -0.000288 | 0.002059 | 0.002346 |
