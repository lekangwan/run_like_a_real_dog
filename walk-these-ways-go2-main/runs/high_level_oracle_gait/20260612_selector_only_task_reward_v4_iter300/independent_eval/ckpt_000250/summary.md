# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000250.pt`
- checkpoint_iteration: 250

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.334 | 0.182 | 0.350 | 0.134 | 0.223 | 4.47 | 0.202 | 0.807 | 0.832 | 0.779 | 0.742 | 0.099 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.310 | 0.190 | 0.343 | 0.157 | 0.224 | 4.45 | 0.220 | 0.784 | 0.817 | 0.725 | 0.750 | 0.100 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.270 | 0.228 | 0.306 | 0.195 | 0.227 | 4.39 | 0.245 | 0.751 | 0.797 | 0.643 | 0.751 | 0.100 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.242 | 0.266 | 0.266 | 0.226 | 0.231 | 4.32 | 0.556 | 0.517 | 0.716 | 0.676 | 0.746 | 0.100 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.176 | 0.298 | 0.237 | 0.288 | 0.230 | 4.33 | 1.062 | 0.171 | 0.597 | 0.589 | 0.763 | 0.101 | 0.000 |
