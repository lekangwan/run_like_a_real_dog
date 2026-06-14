# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.440, trotting=0.433, pronking=0.089, bounding=0.038

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.530 | 0.607 | 0.495 | 0.028 | 1.383 | 0.057 | 1.689 | 204.517 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.529 | 0.628 | 0.491 | 0.023 | 1.439 | 0.057 | 1.506 | 200.898 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=-0.038 |
| 3 | pronking | 0.481 | 0.654 | 0.577 | 0.022 | 1.592 | 0.067 | 1.900 | 221.815 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 4 | bounding | 0.456 | 0.597 | 0.608 | 0.024 | 1.383 | 0.065 | 1.726 | 240.791 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.038 |
