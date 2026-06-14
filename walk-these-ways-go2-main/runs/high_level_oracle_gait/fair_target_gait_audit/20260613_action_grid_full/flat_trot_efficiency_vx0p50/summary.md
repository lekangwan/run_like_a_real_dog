# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.427, trotting=0.395, pacing=0.132, bounding=0.046

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.917 | 0.723 | 0.066 | 0.007 | 1.990 | 0.121 | 1.463 | 252.866 | f=2.61, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.915 | 0.782 | 0.056 | 0.007 | 1.154 | 0.205 | 0.912 | 130.257 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.882 | 0.713 | 0.058 | 0.006 | 1.282 | 0.140 | 0.976 | 195.617 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | bounding | 0.851 | 0.746 | 0.142 | 0.005 | 1.356 | 0.147 | 0.984 | 239.833 | f=2.60, d=0.50, foot=0.090, width=0.380, pitch=0.000 |
