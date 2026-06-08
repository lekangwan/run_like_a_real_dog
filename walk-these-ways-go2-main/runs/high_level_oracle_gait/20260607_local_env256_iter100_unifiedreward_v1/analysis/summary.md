# Oracle Training Metrics Summary

- run_dir: `runs/high_level_oracle_gait/20260607_local_env256_iter100_unifiedreward_v1`
- rows: 100
- iterations: 0 to 99

## Early vs Late

| metric | early | late | delta |
|---|---:|---:|---:|
| reward | 0.657575 | 0.659051 | 0.001476 |
| weighted_metric_reward | 0.657575 | 0.659051 | 0.001476 |
| vx_err | 0.493363 | 0.495248 | 0.001885 |
| score_progress | 0.554526 | 0.553843 | -0.000683 |
| score_clearance | 0.700460 | 0.713292 | 0.012831 |
| score_action_smoothness | 0.139197 | 0.116873 | -0.022324 |
| score_action_magnitude | 0.696048 | 0.617777 | -0.078271 |
| score_action_boundary_margin | 0.999441 | 0.998542 | -0.000899 |
| action_clip_rate | 0.115337 | 0.181255 | 0.065918 |
| footswing_height_mean | 0.093530 | 0.095065 | 0.001535 |
| stance_width_mean | 0.349976 | 0.351496 | 0.001519 |
| body_pitch_mean | 0.006160 | 0.012483 | 0.006323 |

## Per-Task Action Health

| task | clip early | clip late | footswing early | footswing late |
|---|---:|---:|---:|---:|
| flat_trot_efficiency | 0.107091 | 0.165805 | 0.094501 | 0.093339 |
| ramp_up_trot_robustness | 0.109841 | 0.168958 | 0.094426 | 0.093725 |
| rough_slope_trot_robustness | 0.109951 | 0.177929 | 0.094420 | 0.094604 |
| push_lateral_pace_recovery | 0.119424 | 0.187966 | 0.094170 | 0.095450 |
| stepping_stones_easy_bound_highspeed | 0.130539 | 0.205919 | 0.093794 | 0.098124 |

## Target Gait Ratios

| task | target ratio early | target ratio late | delta |
|---|---:|---:|---:|
| flat_trot_efficiency | 0.237500 | 0.282272 | 0.044772 |
| ramp_up_trot_robustness | 0.234436 | 0.278615 | 0.044179 |
| rough_slope_trot_robustness | 0.234375 | 0.265686 | 0.031311 |
| push_lateral_pace_recovery | 0.265870 | 0.278676 | 0.012806 |
| stepping_stones_easy_bound_highspeed | 0.246814 | 0.169853 | -0.076961 |

## Baseline Late vs Current Late

| metric | baseline late | current late | delta |
|---|---:|---:|---:|
| reward | 0.660999 | 0.659051 | -0.001948 |
| weighted_metric_reward | 0.660999 | 0.659051 | -0.001948 |
| vx_err | 0.525485 | 0.495248 | -0.030237 |
| score_progress | 0.526221 | 0.553843 | 0.027622 |
| score_clearance | 0.810095 | 0.713292 | -0.096803 |
| score_action_smoothness | 0.110106 | 0.116873 | 0.006767 |
| score_action_magnitude | 0.668940 | 0.617777 | -0.051163 |
| action_clip_rate | 0.186968 | 0.181255 | -0.005713 |
| footswing_height_mean | 0.102918 | 0.095065 | -0.007854 |
| stance_width_mean | 0.350661 | 0.351496 | 0.000835 |
| body_pitch_mean | 0.003170 | 0.012483 | 0.009312 |
