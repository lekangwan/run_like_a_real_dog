# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.733, pronking=0.115, pacing=0.103, bounding=0.049

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.743 | 0.705 | 0.344 | 0.025 | 1.456 | 0.149 | 1.511 | 184.199 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| 2 | pronking | 0.688 | 0.671 | 0.455 | 0.026 | 1.467 | 0.147 | 1.603 | 190.859 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 3 | pacing | 0.685 | 0.649 | 0.388 | 0.023 | 1.216 | 0.126 | 1.527 | 189.524 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=-0.038 |
| 4 | bounding | 0.663 | 0.648 | 0.390 | 0.027 | 1.419 | 0.145 | 1.799 | 214.598 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=-0.038 |
