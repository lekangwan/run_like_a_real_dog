# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## push_lateral_pace_recovery vx=1.20

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.528, trotting=0.284, bounding=0.121, pacing=0.067

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.733 | 0.747 | 0.170 | 0.016 | 1.205 | 0.128 | 1.616 | 183.665 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.000 |
| 2 | trotting | 0.714 | 0.744 | 0.162 | 0.018 | 1.034 | 0.199 | 1.101 | 140.352 | f=3.39, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | bounding | 0.689 | 0.723 | 0.191 | 0.019 | 1.137 | 0.159 | 1.401 | 191.320 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | pacing | 0.671 | 0.681 | 0.168 | 0.017 | 1.528 | 0.132 | 1.340 | 169.733 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
