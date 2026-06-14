# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.473, trotting=0.342, pacing=0.146, bounding=0.039

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.827 | 0.071 | 0.007 | 1.582 | 0.035 | 1.450 | 252.716 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.882 | 0.746 | 0.068 | 0.007 | 1.817 | 0.069 | 0.896 | 162.567 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.856 | 0.727 | 0.097 | 0.009 | 1.330 | 0.046 | 0.776 | 248.679 | f=2.50, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.817 | 0.697 | 0.109 | 0.006 | 1.640 | 0.077 | 1.057 | 239.962 | f=3.40, d=0.50, foot=0.090, width=0.380, pitch=0.040 |
