# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.423, trotting=0.342, pacing=0.204, bounding=0.031

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.799 | 0.085 | 0.007 | 1.219 | 0.053 | 1.187 | 239.557 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.885 | 0.751 | 0.069 | 0.009 | 1.657 | 0.065 | 0.957 | 166.838 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.870 | 0.766 | 0.093 | 0.008 | 1.226 | 0.015 | 0.603 | 222.966 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.813 | 0.713 | 0.146 | 0.008 | 1.847 | 0.059 | 1.119 | 258.957 | f=2.61, d=0.50, foot=0.090, width=0.341, pitch=0.000 |
