# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260610_rma_task_reward_v4/checkpoints/high_level_000099.pt`
- checkpoint_iteration: 99

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.251 | 0.359 | 0.208 | 0.182 | 0.241 | 4.13 | 0.177 | 0.833 | 0.831 | 0.775 | 0.697 | 0.096 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.255 | 0.345 | 0.207 | 0.192 | 0.241 | 4.13 | 0.205 | 0.797 | 0.816 | 0.716 | 0.705 | 0.096 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.241 | 0.345 | 0.234 | 0.180 | 0.240 | 4.16 | 0.237 | 0.758 | 0.796 | 0.621 | 0.717 | 0.097 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.163 | 0.334 | 0.269 | 0.234 | 0.238 | 4.18 | 0.573 | 0.504 | 0.712 | 0.661 | 0.737 | 0.099 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.119 | 0.318 | 0.275 | 0.288 | 0.235 | 4.24 | 1.080 | 0.160 | 0.595 | 0.574 | 0.767 | 0.101 | 0.000 |
