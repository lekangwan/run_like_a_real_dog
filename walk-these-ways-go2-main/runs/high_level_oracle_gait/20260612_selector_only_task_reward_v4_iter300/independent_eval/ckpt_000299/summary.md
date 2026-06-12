# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000299.pt`
- checkpoint_iteration: 299

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.349 | 0.147 | 0.396 | 0.108 | 0.220 | 4.53 | 0.208 | 0.802 | 0.832 | 0.773 | 0.752 | 0.100 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.316 | 0.175 | 0.387 | 0.122 | 0.221 | 4.50 | 0.219 | 0.785 | 0.817 | 0.713 | 0.754 | 0.100 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.272 | 0.226 | 0.343 | 0.158 | 0.225 | 4.43 | 0.246 | 0.751 | 0.796 | 0.639 | 0.751 | 0.100 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.262 | 0.234 | 0.321 | 0.183 | 0.230 | 4.34 | 0.566 | 0.507 | 0.714 | 0.660 | 0.752 | 0.100 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.225 | 0.243 | 0.290 | 0.242 | 0.233 | 4.27 | 1.055 | 0.177 | 0.601 | 0.585 | 0.766 | 0.101 | 0.000 |
