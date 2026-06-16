# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.715, pacing=0.247, pronking=0.024, bounding=0.013

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.543 | 0.647 | 0.493 | 0.024 | 1.352 | 0.072 | 1.559 | 189.164 | f=2.62, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | pacing | 0.511 | 0.619 | 0.541 | 0.023 | 1.366 | 0.070 | 1.646 | 211.752 | f=2.50, d=0.50, foot=0.120, width=0.342, pitch=0.000 |
| 3 | pronking | 0.441 | 0.644 | 0.649 | 0.024 | 1.293 | 0.073 | 1.634 | 214.207 | f=3.00, d=0.50, foot=0.080, width=0.368, pitch=-0.038 |
| 4 | bounding | 0.424 | 0.603 | 0.641 | 0.017 | 1.657 | 0.064 | 1.655 | 243.284 | f=3.00, d=0.50, foot=0.120, width=0.380, pitch=-0.038 |
