# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000100.pt`
- checkpoint_iteration: 100

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.250 | 0.271 | 0.306 | 0.172 | 0.234 | 4.26 | 0.177 | 0.836 | 0.828 | 0.790 | 0.739 | 0.099 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.245 | 0.266 | 0.319 | 0.170 | 0.235 | 4.23 | 0.201 | 0.803 | 0.814 | 0.729 | 0.745 | 0.100 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.275 | 0.263 | 0.290 | 0.172 | 0.233 | 4.28 | 0.239 | 0.757 | 0.792 | 0.635 | 0.731 | 0.098 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.205 | 0.275 | 0.294 | 0.226 | 0.236 | 4.21 | 0.557 | 0.519 | 0.712 | 0.672 | 0.760 | 0.101 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.182 | 0.257 | 0.308 | 0.253 | 0.236 | 4.22 | 1.058 | 0.179 | 0.596 | 0.580 | 0.781 | 0.102 | 0.000 |
