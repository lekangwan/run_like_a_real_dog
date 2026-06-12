# Selector-Only Task-Onehot Checkpoint Trend

- run_dir: `runs/high_level_oracle_gait/20260612_selector_only_task_reward_v4_iter300`
- eval: independent single-task quick eval
- checkpoints: 50, 100, 150, 200, 250, 299
- mode: selector-only, task-onehot, continuous residual fixed

## Target Gait Ratio Trend

| checkpoint | flat trot | ramp trot | rough trot | push pace | stones bound | mean target | vx_err mean | done_rate mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.247 | 0.251 | 0.250 | 0.183 | 0.371 | 0.260 | 0.441 | 0.018 |
| 100 | 0.271 | 0.266 | 0.263 | 0.226 | 0.308 | 0.267 | 0.447 | 0.018 |
| 150 | 0.267 | 0.275 | 0.267 | 0.205 | 0.315 | 0.266 | 0.447 | 0.018 |
| 200 | 0.263 | 0.276 | 0.294 | 0.187 | 0.300 | 0.264 | 0.449 | 0.018 |
| 250 | 0.182 | 0.190 | 0.228 | 0.226 | 0.237 | 0.213 | 0.457 | 0.017 |
| 299 | 0.147 | 0.175 | 0.226 | 0.183 | 0.290 | 0.204 | 0.459 | 0.017 |

## Readout

The weak target-gait signal at 100-200 iterations does not amplify with longer selector-only training. By checkpoint 250 and 299, flat/ramp trot collapse sharply, push pace never becomes dominant, and stones bound remains inconsistent.

This supports moving to a stronger selector training signal, such as a soft gait prior, rather than continuing selector-only reward-v4 training.
