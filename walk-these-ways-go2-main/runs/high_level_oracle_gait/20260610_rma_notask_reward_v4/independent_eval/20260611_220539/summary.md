# Independent High-Level Policy Evaluation

- checkpoint: `runs/high_level_oracle_gait/20260610_rma_notask_reward_v4/checkpoints/high_level_000099.pt`
- checkpoint_iteration: 99

| task | vx | target | pronk | trot | bound | pace | switch | dwell | vx_err | progress | slip | orientation | clearance | foot | clip |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_trot_efficiency | 1.00 | trotting | 0.051 | 0.419 | 0.297 | 0.233 | 0.235 | 4.24 | 0.185 | 0.820 | 0.820 | 0.792 | 0.783 | 0.103 | 0.000 |
| ramp_up_trot_robustness | 1.00 | trotting | 0.053 | 0.422 | 0.287 | 0.239 | 0.233 | 4.28 | 0.219 | 0.780 | 0.805 | 0.736 | 0.780 | 0.102 | 0.000 |
| rough_slope_trot_robustness | 1.00 | trotting | 0.061 | 0.421 | 0.290 | 0.228 | 0.234 | 4.26 | 0.240 | 0.755 | 0.793 | 0.650 | 0.777 | 0.102 | 0.000 |
| push_lateral_pace_recovery | 1.50 | pacing | 0.060 | 0.393 | 0.275 | 0.271 | 0.235 | 4.23 | 0.561 | 0.509 | 0.714 | 0.675 | 0.787 | 0.103 | 0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | bounding | 0.072 | 0.377 | 0.253 | 0.298 | 0.235 | 4.23 | 1.033 | 0.186 | 0.600 | 0.603 | 0.796 | 0.104 | 0.000 |
