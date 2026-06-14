# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## push_lateral_pace_recovery vx=1.50

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.465, trotting=0.330, bounding=0.105, pacing=0.100

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.701 | 0.752 | 0.274 | 0.017 | 1.217 | 0.151 | 1.399 | 165.605 | f=3.00, d=0.50, foot=0.051, width=0.291, pitch=0.000 |
| 2 | trotting | 0.691 | 0.717 | 0.236 | 0.020 | 1.286 | 0.203 | 1.203 | 148.354 | f=3.38, d=0.50, foot=0.051, width=0.292, pitch=0.038 |
| 3 | bounding | 0.656 | 0.759 | 0.256 | 0.020 | 1.300 | 0.143 | 1.496 | 198.216 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | pacing | 0.655 | 0.684 | 0.230 | 0.015 | 1.316 | 0.134 | 1.405 | 170.643 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.038 |
