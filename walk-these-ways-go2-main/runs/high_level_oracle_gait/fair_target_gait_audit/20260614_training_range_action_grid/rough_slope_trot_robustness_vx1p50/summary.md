# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.384, trotting=0.288, pacing=0.212, bounding=0.116

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.670 | 0.675 | 0.313 | 0.020 | 1.539 | 0.073 | 1.378 | 173.789 | f=3.39, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 2 | trotting | 0.661 | 0.677 | 0.323 | 0.023 | 1.034 | 0.068 | 1.258 | 173.285 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=0.038 |
| 3 | pacing | 0.652 | 0.684 | 0.293 | 0.022 | 1.336 | 0.058 | 1.429 | 179.268 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.634 | 0.647 | 0.301 | 0.018 | 1.392 | 0.057 | 1.537 | 202.107 | f=3.39, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
