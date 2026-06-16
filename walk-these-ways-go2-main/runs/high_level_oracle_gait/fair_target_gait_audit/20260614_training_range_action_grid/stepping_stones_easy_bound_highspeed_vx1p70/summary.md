# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## stepping_stones_easy_bound_highspeed vx=1.70

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.319, pronking=0.294, trotting=0.204, bounding=0.183

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.595 | 0.664 | 0.462 | 0.019 | 1.388 | 0.113 | 1.448 | 222.920 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 2 | pronking | 0.593 | 0.684 | 0.470 | 0.019 | 1.608 | 0.117 | 1.600 | 223.382 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.000 |
| 3 | trotting | 0.582 | 0.675 | 0.517 | 0.021 | 1.309 | 0.114 | 1.430 | 222.927 | f=2.61, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 4 | bounding | 0.579 | 0.660 | 0.493 | 0.021 | 1.578 | 0.110 | 1.469 | 238.862 | f=2.61, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
