# Fair Task-Speed Gait Decision Analysis

This analysis starts from `best_by_task_speed_gait.csv`: each gait has already been given an equal continuous-parameter search budget, and only that gait family's best setting is compared against other gait families.

Margin labels: `<0.01` tie/noise, `0.01-0.03` weak advantage, `>=0.03` clear advantage. These are decision aids, not physical laws.

## Decision Table

| task | vx | top | second | margin | label | recommended target form | top prob | top params |
|---|---:|---|---|---:|---|---|---:|---|
| flat_trot_efficiency | 0.50 | pronking | trotting | 0.005 | tie_or_noise | soft_tie | 0.445 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| flat_trot_efficiency | 1.00 | pronking | trotting | 0.002 | tie_or_noise | soft_tie | 0.428 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| flat_trot_efficiency | 1.50 | trotting | pronking | 0.008 | tie_or_noise | soft_tie | 0.475 | f=2.61, d=0.50, foot=0.109, width=0.330, pitch=0.000 |
| flat_trot_efficiency | 2.00 | trotting | pronking | 0.055 | clear_advantage | hard_or_sharp_soft | 0.733 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| push_lateral_pace_recovery | 1.20 | pronking | trotting | 0.019 | weak_advantage | soft_preference | 0.528 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.000 |
| push_lateral_pace_recovery | 1.50 | pronking | trotting | 0.012 | weak_advantage | soft_preference | 0.522 | f=2.62, d=0.50, foot=0.109, width=0.368, pitch=0.000 |
| push_lateral_pace_recovery | 1.80 | pronking | trotting | 0.003 | tie_or_noise | soft_tie | 0.416 | f=2.62, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| ramp_up_trot_robustness | 0.50 | pronking | trotting | 0.006 | tie_or_noise | soft_tie | 0.423 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| ramp_up_trot_robustness | 1.00 | pronking | trotting | 0.009 | tie_or_noise | soft_tie | 0.488 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| ramp_up_trot_robustness | 1.50 | pronking | trotting | 0.013 | weak_advantage | soft_preference | 0.522 | f=3.00, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| ramp_up_trot_robustness | 2.00 | pronking | trotting | 0.020 | weak_advantage | soft_preference | 0.606 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| rough_slope_trot_robustness | 0.50 | pronking | trotting | 0.011 | weak_advantage | soft_preference | 0.513 | f=3.39, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| rough_slope_trot_robustness | 1.00 | pronking | trotting | 0.010 | tie_or_noise | soft_tie | 0.493 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| rough_slope_trot_robustness | 1.50 | pronking | trotting | 0.009 | tie_or_noise | soft_tie | 0.384 | f=3.39, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| rough_slope_trot_robustness | 2.00 | trotting | pacing | 0.032 | clear_advantage | hard_or_sharp_soft | 0.715 | f=2.62, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| stepping_stones_easy_bound_highspeed | 1.70 | pacing | pronking | 0.002 | tie_or_noise | soft_tie | 0.319 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| stepping_stones_easy_bound_highspeed | 2.00 | pacing | trotting | 0.028 | weak_advantage | soft_preference | 0.578 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |

## Practical Readout

- The useful object is not a single gait label; it is the per-task-speed table of each gait family at its own best continuous parameters.
- Low-margin rows should remain soft targets or diagnostic comparisons, because the top gait is not robustly separated.
- Clearer rows can become sharp soft targets or hard labels only if raw metrics and Pareto trade-offs are acceptable.
- Before changing PPO reward, compare the neutral ranking against live weighted reward; disagreements mean reward-only training will optimize a different objective.