# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.445, trotting=0.374, pacing=0.131, bounding=0.050

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.919 | 0.722 | 0.072 | 0.010 | 1.212 | 0.115 | 1.197 | 246.220 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.914 | 0.766 | 0.065 | 0.008 | 1.593 | 0.199 | 0.928 | 133.956 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.883 | 0.720 | 0.057 | 0.008 | 1.155 | 0.152 | 1.212 | 221.696 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.854 | 0.731 | 0.149 | 0.006 | 1.651 | 0.159 | 1.052 | 232.432 | f=2.61, d=0.50, foot=0.090, width=0.341, pitch=0.039 |
