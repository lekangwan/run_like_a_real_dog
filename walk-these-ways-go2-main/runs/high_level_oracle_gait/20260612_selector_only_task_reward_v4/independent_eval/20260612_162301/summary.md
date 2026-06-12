# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4/checkpoints/high_level_000099.pt`
- checkpoint_iteration: 99

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.176 | 0.298 | 0.276 | 0.250 | 0.231 | 4.31 | 0.184 | 0.822 | 0.821 | 0.783 | 0.763 | 0.101 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.169 | 0.309 | 0.250 | 0.272 | 0.232 | 4.30 | 0.213 | 0.785 | 0.804 | 0.727 | 0.761 | 0.101 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.178 | 0.296 | 0.269 | 0.258 | 0.232 | 4.29 | 0.245 | 0.747 | 0.789 | 0.641 | 0.763 | 0.101 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.179 | 0.268 | 0.273 | 0.280 | 0.234 | 4.26 | 0.561 | 0.513 | 0.714 | 0.669 | 0.777 | 0.102 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.177 | 0.264 | 0.272 | 0.287 | 0.233 | 4.28 | 1.065 | 0.174 | 0.600 | 0.578 | 0.779 | 0.102 | 0.000 |
