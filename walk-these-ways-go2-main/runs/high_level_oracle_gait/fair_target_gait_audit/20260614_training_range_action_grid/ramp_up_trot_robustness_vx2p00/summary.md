# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.606, trotting=0.315, pacing=0.067, bounding=0.012

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.655 | 0.713 | 0.421 | 0.023 | 1.504 | 0.056 | 1.883 | 198.491 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | trotting | 0.635 | 0.699 | 0.446 | 0.022 | 1.421 | 0.050 | 1.453 | 179.367 | f=3.00, d=0.50, foot=0.051, width=0.292, pitch=0.000 |
| 3 | pacing | 0.589 | 0.649 | 0.485 | 0.029 | 1.387 | 0.037 | 1.626 | 204.289 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.537 | 0.640 | 0.496 | 0.024 | 1.364 | 0.039 | 1.823 | 208.179 | f=3.00, d=0.50, foot=0.091, width=0.342, pitch=-0.038 |
