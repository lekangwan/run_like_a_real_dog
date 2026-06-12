# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000200.pt`
- checkpoint_iteration: 200

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.257 | 0.263 | 0.356 | 0.124 | 0.225 | 4.43 | 0.187 | 0.825 | 0.832 | 0.783 | 0.740 | 0.099 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.250 | 0.276 | 0.352 | 0.122 | 0.226 | 4.41 | 0.213 | 0.790 | 0.817 | 0.723 | 0.737 | 0.099 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.240 | 0.294 | 0.317 | 0.149 | 0.229 | 4.34 | 0.238 | 0.760 | 0.800 | 0.649 | 0.733 | 0.099 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.224 | 0.293 | 0.296 | 0.187 | 0.232 | 4.28 | 0.551 | 0.521 | 0.717 | 0.673 | 0.741 | 0.099 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.193 | 0.286 | 0.300 | 0.220 | 0.233 | 4.26 | 1.056 | 0.177 | 0.600 | 0.585 | 0.760 | 0.101 | 0.000 |
