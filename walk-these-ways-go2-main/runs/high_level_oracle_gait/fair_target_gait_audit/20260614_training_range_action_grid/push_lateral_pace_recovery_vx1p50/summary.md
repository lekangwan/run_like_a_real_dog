# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## push_lateral_pace_recovery vx=1.50

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.522, trotting=0.347, bounding=0.066, pacing=0.065

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.710 | 0.740 | 0.249 | 0.019 | 1.148 | 0.150 | 1.618 | 187.235 | f=2.62, d=0.50, foot=0.109, width=0.368, pitch=0.000 |
| 2 | trotting | 0.698 | 0.735 | 0.217 | 0.022 | 1.189 | 0.169 | 1.278 | 161.357 | f=3.38, d=0.50, foot=0.080, width=0.292, pitch=0.038 |
| 3 | bounding | 0.648 | 0.752 | 0.262 | 0.018 | 1.283 | 0.140 | 1.538 | 198.087 | f=3.00, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | pacing | 0.648 | 0.698 | 0.264 | 0.022 | 1.122 | 0.123 | 1.354 | 177.114 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.038 |
