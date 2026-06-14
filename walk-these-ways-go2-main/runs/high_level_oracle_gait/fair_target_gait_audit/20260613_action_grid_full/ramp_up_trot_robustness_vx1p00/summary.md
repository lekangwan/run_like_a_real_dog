# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.533, trotting=0.318, pacing=0.123, bounding=0.026

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.863 | 0.747 | 0.126 | 0.013 | 1.310 | 0.064 | 1.265 | 161.539 | f=3.39, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 2 | trotting | 0.848 | 0.733 | 0.145 | 0.014 | 1.521 | 0.060 | 1.097 | 146.929 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.820 | 0.686 | 0.145 | 0.017 | 1.592 | 0.050 | 1.194 | 163.096 | f=2.89, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
| 4 | bounding | 0.773 | 0.708 | 0.168 | 0.012 | 1.673 | 0.059 | 1.195 | 187.050 | f=3.39, d=0.50, foot=0.091, width=0.380, pitch=0.000 |
