# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.688, pronking=0.139, pacing=0.109, bounding=0.063

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.739 | 0.707 | 0.339 | 0.023 | 1.382 | 0.151 | 1.487 | 181.087 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | pronking | 0.692 | 0.796 | 0.452 | 0.019 | 1.366 | 0.169 | 1.647 | 180.636 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.000 |
| 3 | pacing | 0.684 | 0.671 | 0.382 | 0.026 | 1.390 | 0.127 | 1.543 | 186.463 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.668 | 0.657 | 0.392 | 0.026 | 1.572 | 0.139 | 1.670 | 203.551 | f=3.38, d=0.50, foot=0.091, width=0.380, pitch=0.000 |
