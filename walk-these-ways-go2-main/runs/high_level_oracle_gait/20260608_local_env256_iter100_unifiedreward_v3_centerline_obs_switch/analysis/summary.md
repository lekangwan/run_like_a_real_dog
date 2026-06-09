# Oracle Training Metrics Summary

- run_dir: `runs/high_level_oracle_gait/20260608_local_env256_iter100_unifiedreward_v3_centerline_obs_switch`
- rows: 100
- iterations: 0 to 99

## Early vs Late

| metric | early | late | delta |
|---|---:|---:|---:|
| reward | 0.664725 | 0.660180 | -0.004545 |
| weighted_metric_reward | 0.664725 | 0.660180 | -0.004545 |
| vx_err | 0.454625 | 0.448757 | -0.005869 |
| lateral_position_penalty | 3.150966 | 3.368665 | 0.217699 |
| gait_switch_penalty | 0.118402 | 0.114752 | -0.003650 |
| gait_switch_rate | 0.235257 | 0.231426 | -0.003831 |
| score_progress | 0.593441 | 0.596383 | 0.002942 |
| score_clearance | 0.738931 | 0.759771 | 0.020840 |
| score_gait_stability | 0.795244 | 0.801556 | 0.006312 |
| score_action_smoothness | 0.417694 | 0.368642 | -0.049052 |
| score_action_magnitude | 0.734658 | 0.679599 | -0.055059 |
| score_action_boundary_margin | 0.999913 | 0.999427 | -0.000486 |
| action_clip_rate | 0.000000 | 0.000007 | 0.000007 |
| footswing_height_mean | 0.099206 | 0.100650 | 0.001444 |
| stance_width_mean | 0.354145 | 0.354360 | 0.000215 |
| body_pitch_mean | -0.007715 | -0.001284 | 0.006431 |

## Per-Task Action Health

| task | clip early | clip late | switch early | switch late | footswing early | footswing late |
|---|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 0.000000 | 0.000000 | 0.236538 | 0.231886 | 0.099578 | 0.099806 |
| ramp_up_trot_robustness | 0.000000 | 0.000000 | 0.235041 | 0.231183 | 0.099486 | 0.100051 |
| rough_slope_trot_robustness | 0.000000 | 0.000012 | 0.233017 | 0.230993 | 0.099201 | 0.101152 |
| push_lateral_pace_recovery | 0.000000 | 0.000025 | 0.237381 | 0.232321 | 0.099029 | 0.101301 |
| stepping_stones_easy_bound_highspeed | 0.000000 | 0.000000 | 0.234282 | 0.230740 | 0.098269 | 0.101618 |

## Target Gait Ratios

| task | target ratio early | target ratio late | delta |
|---|---:|---:|---:|
| flat_trot_efficiency | 0.242067 | 0.346695 | 0.104627 |
| ramp_up_trot_robustness | 0.245221 | 0.349081 | 0.103860 |
| rough_slope_trot_robustness | 0.245711 | 0.336949 | 0.091238 |
| push_lateral_pace_recovery | 0.287071 | 0.341054 | 0.053983 |
| stepping_stones_easy_bound_highspeed | 0.261458 | 0.179902 | -0.081556 |

## Baseline Late vs Current Late

| metric | baseline late | current late | delta |
|---|---:|---:|---:|
| reward | 0.599111 | 0.660180 | 0.061069 |
| weighted_metric_reward | 0.599111 | 0.660180 | 0.061069 |
| vx_err | 0.539050 | 0.448757 | -0.090294 |
| lateral_position_penalty | 3.508427 | 3.368665 | -0.139761 |
| score_progress | 0.512309 | 0.596383 | 0.084075 |
| score_clearance | 0.749775 | 0.759771 | 0.009996 |
| score_action_smoothness | 0.107945 | 0.368642 | 0.260698 |
| score_action_magnitude | 0.682743 | 0.679599 | -0.003144 |
| score_action_boundary_margin | 0.999736 | 0.999427 | -0.000309 |
| action_clip_rate | 0.171458 | 0.000007 | -0.171450 |
| footswing_height_mean | 0.097279 | 0.100650 | 0.003371 |
| stance_width_mean | 0.346362 | 0.354360 | 0.007997 |
| body_pitch_mean | -0.003640 | -0.001284 | 0.002357 |
