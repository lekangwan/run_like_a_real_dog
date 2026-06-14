# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.482, pronking=0.363, pacing=0.101, bounding=0.055

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.910 | 0.785 | 0.056 | 0.000 | 0.860 | 0.148 | 0.621 | 126.006 | f=3.00, d=0.50, foot=0.080, width=0.290, pitch=0.040 |
| 2 | pronking | 0.902 | 0.718 | 0.111 | 0.000 | 0.898 | 0.144 | 1.701 | 165.449 | f=2.60, d=0.50, foot=0.080, width=0.290, pitch=0.040 |
| 3 | pacing | 0.864 | 0.758 | 0.108 | 0.000 | 0.513 | 0.033 | 2.078 | 144.502 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.845 | 0.737 | 0.117 | 0.000 | 0.936 | 0.138 | 1.748 | 148.463 | f=3.40, d=0.50, foot=0.090, width=0.380, pitch=0.000 |
