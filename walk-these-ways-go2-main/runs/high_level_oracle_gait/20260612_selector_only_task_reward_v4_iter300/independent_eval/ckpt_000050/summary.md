# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000050.pt`
- checkpoint_iteration: 50

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.291 | 0.247 | 0.353 | 0.109 | 0.236 | 4.22 | 0.179 | 0.835 | 0.837 | 0.794 | 0.731 | 0.098 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.286 | 0.251 | 0.342 | 0.121 | 0.237 | 4.21 | 0.208 | 0.801 | 0.819 | 0.742 | 0.732 | 0.099 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.291 | 0.250 | 0.316 | 0.142 | 0.237 | 4.20 | 0.234 | 0.765 | 0.798 | 0.647 | 0.729 | 0.098 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.252 | 0.219 | 0.346 | 0.183 | 0.235 | 4.24 | 0.543 | 0.536 | 0.719 | 0.674 | 0.764 | 0.101 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.225 | 0.189 | 0.371 | 0.215 | 0.233 | 4.28 | 1.041 | 0.193 | 0.601 | 0.573 | 0.793 | 0.103 | 0.000 |
