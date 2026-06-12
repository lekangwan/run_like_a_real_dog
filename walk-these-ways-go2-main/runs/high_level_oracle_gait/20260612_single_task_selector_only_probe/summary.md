# Single-Task Selector-Only Probe Summary

- purpose: test whether reward v4 can train gait selector when task-gradient competition is removed
- mode: selector-only, task-onehot, continuous residual fixed to zero
- training: 100 iterations per task
- eval: independent single-task rollout from `high_level_000099.pt`

## Results

| task | target | target ratio | dominant | dominant ratio | gait ratios p/t/b/pa | vx_err | done_rate |
|---|---|---:|---|---:|---:|---:|---:|
| flat_trot_efficiency | trotting | 0.298 | bound | 0.367 | 0.16/0.30/0.37/0.17 | 0.181 | 0.017 |
| ramp_up_trot_robustness | trotting | 0.237 | pronk | 0.260 | 0.26/0.24/0.25/0.25 | 0.208 | 0.017 |
| rough_slope_trot_robustness | trotting | 0.311 | bound | 0.336 | 0.23/0.31/0.34/0.13 | 0.245 | 0.017 |
| push_lateral_pace_recovery | pacing | 0.248 | trot | 0.280 | 0.22/0.28/0.26/0.25 | 0.554 | 0.020 |
| stepping_stones_easy_bound_highspeed | bounding | 0.251 | pace | 0.269 | 0.21/0.27/0.25/0.27 | 1.077 | 0.018 |

## Readout

Single-task training does not make the target gait dominant. This weakens the curriculum-learning hypothesis: removing mixed-task gradient competition is not enough for reward v4 to train the selector.

Recommended next step: move to a soft gait prior or another explicit selector training signal before investing in staged curriculum.
