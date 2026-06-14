# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.449, pronking=0.274, pacing=0.232, bounding=0.045

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.669 | 0.694 | 0.293 | 0.019 | 1.498 | 0.064 | 1.367 | 171.494 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 2 | pronking | 0.654 | 0.689 | 0.319 | 0.018 | 1.384 | 0.068 | 1.567 | 191.382 | f=3.39, d=0.50, foot=0.109, width=0.330, pitch=0.039 |
| 3 | pacing | 0.649 | 0.646 | 0.304 | 0.022 | 1.528 | 0.044 | 1.484 | 181.591 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.600 | 0.652 | 0.349 | 0.022 | 1.398 | 0.061 | 1.583 | 209.680 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
