# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.488, trotting=0.359, pacing=0.120, bounding=0.033

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.854 | 0.751 | 0.117 | 0.018 | 1.518 | 0.051 | 1.352 | 170.669 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 2 | trotting | 0.845 | 0.720 | 0.129 | 0.012 | 1.788 | 0.045 | 1.292 | 164.034 | f=2.61, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.812 | 0.826 | 0.150 | 0.016 | 1.521 | 0.041 | 1.183 | 174.451 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.773 | 0.680 | 0.171 | 0.014 | 1.798 | 0.044 | 1.320 | 221.114 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
