# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## push_lateral_pace_recovery vx=1.80

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.416, trotting=0.374, bounding=0.133, pacing=0.077

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.662 | 0.725 | 0.366 | 0.024 | 1.275 | 0.179 | 1.699 | 186.419 | f=2.62, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| 2 | trotting | 0.658 | 0.740 | 0.302 | 0.023 | 1.107 | 0.164 | 1.458 | 171.536 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=0.000 |
| 3 | bounding | 0.627 | 0.690 | 0.331 | 0.026 | 1.220 | 0.152 | 1.547 | 202.126 | f=3.38, d=0.50, foot=0.091, width=0.342, pitch=-0.038 |
| 4 | pacing | 0.611 | 0.694 | 0.322 | 0.021 | 1.301 | 0.124 | 1.421 | 178.240 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.000 |
