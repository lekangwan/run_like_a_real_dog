# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.493, trotting=0.356, pacing=0.113, bounding=0.039

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.783 | 0.773 | 0.171 | 0.015 | 1.675 | 0.056 | 1.403 | 177.300 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.773 | 0.700 | 0.165 | 0.018 | 1.339 | 0.058 | 1.241 | 173.124 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.739 | 0.650 | 0.175 | 0.016 | 1.563 | 0.068 | 1.304 | 172.418 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.707 | 0.656 | 0.185 | 0.019 | 1.267 | 0.070 | 1.057 | 177.217 | f=3.39, d=0.50, foot=0.091, width=0.380, pitch=0.039 |
