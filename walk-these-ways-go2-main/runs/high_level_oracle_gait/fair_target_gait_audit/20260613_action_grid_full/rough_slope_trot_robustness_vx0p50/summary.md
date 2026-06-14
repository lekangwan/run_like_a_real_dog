# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.521, trotting=0.335, pacing=0.115, bounding=0.029

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.842 | 0.749 | 0.093 | 0.006 | 1.440 | 0.061 | 1.097 | 237.564 | f=3.00, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 2 | trotting | 0.829 | 0.741 | 0.090 | 0.006 | 1.770 | 0.083 | 0.893 | 180.503 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.797 | 0.676 | 0.098 | 0.007 | 1.909 | 0.044 | 0.974 | 193.635 | f=2.50, d=0.50, foot=0.120, width=0.340, pitch=0.040 |
| 4 | bounding | 0.756 | 0.674 | 0.129 | 0.008 | 1.247 | 0.063 | 1.174 | 253.928 | f=3.39, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
