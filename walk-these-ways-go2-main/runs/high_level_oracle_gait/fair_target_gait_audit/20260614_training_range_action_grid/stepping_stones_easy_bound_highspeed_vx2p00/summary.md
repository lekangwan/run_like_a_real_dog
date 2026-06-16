# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## stepping_stones_easy_bound_highspeed vx=2.00

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.578, trotting=0.224, bounding=0.121, pronking=0.076

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.566 | 0.716 | 0.587 | 0.025 | 1.770 | 0.102 | 1.374 | 211.346 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.537 | 0.676 | 0.688 | 0.020 | 1.320 | 0.113 | 1.489 | 228.843 | f=2.62, d=0.50, foot=0.109, width=0.330, pitch=0.000 |
| 3 | bounding | 0.519 | 0.713 | 0.686 | 0.018 | 1.238 | 0.103 | 1.597 | 260.045 | f=2.62, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | pronking | 0.505 | 0.676 | 0.759 | 0.020 | 1.573 | 0.121 | 1.548 | 261.752 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
