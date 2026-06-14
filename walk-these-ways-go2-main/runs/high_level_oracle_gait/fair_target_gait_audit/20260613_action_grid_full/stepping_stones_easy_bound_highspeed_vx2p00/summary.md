# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## stepping_stones_easy_bound_highspeed vx=2.00

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.442, trotting=0.319, bounding=0.139, pronking=0.099

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.549 | 0.681 | 0.587 | 0.022 | 1.390 | 0.114 | 1.556 | 220.003 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.539 | 0.665 | 0.635 | 0.019 | 1.319 | 0.127 | 1.548 | 234.228 | f=3.00, d=0.50, foot=0.109, width=0.368, pitch=0.038 |
| 3 | bounding | 0.514 | 0.644 | 0.709 | 0.020 | 1.605 | 0.105 | 1.452 | 269.422 | f=2.62, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | pronking | 0.504 | 0.671 | 0.720 | 0.024 | 1.368 | 0.127 | 1.561 | 266.343 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
