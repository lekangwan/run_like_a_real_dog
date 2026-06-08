# Oracle Training Metrics Summary

- run_dir: `runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v2_centerline`
- rows: 100
- iterations: 0 to 99

## Early vs Late

| metric | early | late | delta |
|---|---:|---:|---:|
| reward | 0.617540 | 0.599111 | -0.018428 |
| weighted_metric_reward | 0.617540 | 0.599111 | -0.018428 |
| vx_err | 0.483108 | 0.539050 | 0.055942 |
| lateral_position_penalty | 3.278651 | 3.508427 | 0.229776 |
| score_progress | 0.561038 | 0.512309 | -0.048730 |
| score_clearance | 0.730594 | 0.749775 | 0.019181 |
| score_action_smoothness | 0.136723 | 0.107945 | -0.028778 |
| score_action_magnitude | 0.754421 | 0.682743 | -0.071678 |
| score_action_boundary_margin | 0.999980 | 0.999736 | -0.000244 |
| action_clip_rate | 0.099629 | 0.171458 | 0.071829 |
| footswing_height_mean | 0.096530 | 0.097279 | 0.000749 |
| stance_width_mean | 0.351691 | 0.346362 | -0.005329 |
| body_pitch_mean | -0.005506 | -0.003640 | 0.001866 |

## Per-Task Action Health

| task | clip early | clip late | footswing early | footswing late |
|---|---:|---:|---:|---:|
| flat_trot_efficiency | 0.092800 | 0.155986 | 0.096284 | 0.097076 |
| ramp_up_trot_robustness | 0.094534 | 0.161213 | 0.096441 | 0.096797 |
| rough_slope_trot_robustness | 0.095306 | 0.166434 | 0.096286 | 0.096614 |
| push_lateral_pace_recovery | 0.103272 | 0.177132 | 0.096822 | 0.098607 |
| stepping_stones_easy_bound_highspeed | 0.112365 | 0.196826 | 0.097231 | 0.100360 |

## Target Gait Ratios

| task | target ratio early | target ratio late | delta |
|---|---:|---:|---:|
| flat_trot_efficiency | 0.224940 | 0.249159 | 0.024219 |
| ramp_up_trot_robustness | 0.214890 | 0.255270 | 0.040380 |
| rough_slope_trot_robustness | 0.215441 | 0.258517 | 0.043076 |
| push_lateral_pace_recovery | 0.246507 | 0.245833 | -0.000674 |
| stepping_stones_easy_bound_highspeed | 0.256801 | 0.237071 | -0.019730 |

## Baseline Late vs Current Late

| metric | baseline late | current late | delta |
|---|---:|---:|---:|
| reward | 0.659051 | 0.599111 | -0.059940 |
| weighted_metric_reward | 0.659051 | 0.599111 | -0.059940 |
| vx_err | 0.495248 | 0.539050 | 0.043802 |
| score_progress | 0.553843 | 0.512309 | -0.041534 |
| score_clearance | 0.713292 | 0.749775 | 0.036484 |
| score_action_smoothness | 0.116873 | 0.107945 | -0.008929 |
| score_action_magnitude | 0.617777 | 0.682743 | 0.064966 |
| score_action_boundary_margin | 0.998542 | 0.999736 | 0.001194 |
| action_clip_rate | 0.181255 | 0.171458 | -0.009797 |
| footswing_height_mean | 0.095065 | 0.097279 | 0.002214 |
| stance_width_mean | 0.351496 | 0.346362 | -0.005134 |
| body_pitch_mean | 0.012483 | -0.003640 | -0.016123 |

## Route Test

- route_dir: `runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v2_centerline/route_tests/20260608_222329`
- route steps recorded: 120

| segment | condition | target | steps | reward | vx_err | lateral_offset | done | clip | pronk | trot | bound | pace |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | flat | trotting | 80 | 0.730104 | 0.182418 | 0.414885 | 0.000000 | 0.000000 | 0.212500 | 0.237500 | 0.287500 | 0.262500 |
| 1 | ramp_up | trotting | 40 | 0.670400 | 0.237347 | 2.595236 | 0.025000 | 0.000000 | 0.325000 | 0.250000 | 0.225000 | 0.200000 |
| 2 | rough_slope | trotting | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | push_lateral | pacing | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | stepping_stones_easy | bounding | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
