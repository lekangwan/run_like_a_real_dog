# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.592, pronking=0.202, pacing=0.148, bounding=0.058

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.881 | 0.893 | 0.078 | 0.000 | 0.718 | 0.117 | 1.493 | 138.312 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.000 |
| 2 | pronking | 0.849 | 0.718 | 0.170 | 0.000 | 0.964 | 0.188 | 1.619 | 187.596 | f=3.00, d=0.50, foot=0.110, width=0.290, pitch=0.000 |
| 3 | pacing | 0.840 | 0.670 | 0.131 | 0.000 | 2.022 | 0.138 | 1.642 | 154.966 | f=2.90, d=0.50, foot=0.120, width=0.380, pitch=0.040 |
| 4 | bounding | 0.812 | 0.691 | 0.128 | 0.000 | 2.202 | 0.156 | 2.090 | 172.822 | f=3.40, d=0.50, foot=0.120, width=0.340, pitch=0.000 |
