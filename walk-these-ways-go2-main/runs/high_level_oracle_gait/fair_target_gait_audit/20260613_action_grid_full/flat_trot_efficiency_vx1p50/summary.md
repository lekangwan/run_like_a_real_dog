# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.482, pronking=0.329, pacing=0.125, bounding=0.064

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.828 | 0.723 | 0.202 | 0.016 | 1.550 | 0.150 | 1.340 | 162.839 | f=3.00, d=0.50, foot=0.109, width=0.292, pitch=0.038 |
| 2 | pronking | 0.816 | 0.700 | 0.245 | 0.023 | 1.587 | 0.153 | 1.535 | 181.690 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 3 | pacing | 0.787 | 0.671 | 0.229 | 0.021 | 1.561 | 0.119 | 1.317 | 169.859 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.768 | 0.710 | 0.236 | 0.022 | 1.401 | 0.145 | 1.571 | 196.124 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.000 |
