# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300/checkpoints/high_level_000150.pt`
- checkpoint_iteration: 150

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.264 | 0.267 | 0.324 | 0.144 | 0.229 | 4.34 | 0.182 | 0.829 | 0.831 | 0.786 | 0.734 | 0.099 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.248 | 0.275 | 0.307 | 0.169 | 0.230 | 4.33 | 0.203 | 0.801 | 0.817 | 0.730 | 0.738 | 0.099 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.266 | 0.267 | 0.281 | 0.186 | 0.228 | 4.37 | 0.239 | 0.757 | 0.796 | 0.649 | 0.733 | 0.099 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.209 | 0.291 | 0.294 | 0.205 | 0.233 | 4.27 | 0.555 | 0.522 | 0.715 | 0.673 | 0.750 | 0.100 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.184 | 0.280 | 0.315 | 0.220 | 0.233 | 4.28 | 1.058 | 0.182 | 0.603 | 0.574 | 0.768 | 0.101 | 0.000 |
