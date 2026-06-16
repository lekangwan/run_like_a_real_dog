# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.513, trotting=0.360, pacing=0.097, bounding=0.030

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.842 | 0.745 | 0.085 | 0.006 | 1.507 | 0.067 | 1.058 | 226.497 | f=3.39, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.831 | 0.730 | 0.085 | 0.005 | 1.734 | 0.077 | 0.966 | 160.533 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.791 | 0.683 | 0.098 | 0.008 | 1.773 | 0.024 | 0.764 | 200.622 | f=2.50, d=0.50, foot=0.120, width=0.340, pitch=0.040 |
| 4 | bounding | 0.756 | 0.664 | 0.124 | 0.008 | 1.441 | 0.083 | 0.974 | 234.439 | f=3.39, d=0.50, foot=0.090, width=0.380, pitch=0.039 |
