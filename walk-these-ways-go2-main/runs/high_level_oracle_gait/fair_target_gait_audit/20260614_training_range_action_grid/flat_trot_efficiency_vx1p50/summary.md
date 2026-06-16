# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.475, pronking=0.369, pacing=0.112, bounding=0.044

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.828 | 0.741 | 0.205 | 0.023 | 1.342 | 0.161 | 1.470 | 165.609 | f=2.61, d=0.50, foot=0.109, width=0.330, pitch=0.000 |
| 2 | pronking | 0.820 | 0.682 | 0.245 | 0.018 | 1.653 | 0.141 | 1.640 | 180.742 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 3 | pacing | 0.784 | 0.696 | 0.240 | 0.022 | 1.589 | 0.130 | 1.378 | 179.967 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.756 | 0.705 | 0.243 | 0.018 | 1.626 | 0.154 | 1.499 | 191.935 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.000 |
