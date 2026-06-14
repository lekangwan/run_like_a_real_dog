# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.453, trotting=0.355, pacing=0.134, bounding=0.059

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.928 | 0.725 | 0.074 | 0.000 | 1.011 | 0.163 | 1.138 | 156.797 | f=3.00, d=0.50, foot=0.050, width=0.290, pitch=0.040 |
| 2 | trotting | 0.920 | 0.815 | 0.065 | 0.000 | 1.164 | 0.188 | 1.190 | 117.806 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.000 |
| 3 | pacing | 0.891 | 0.702 | 0.057 | 0.000 | 1.078 | 0.192 | 0.994 | 142.754 | f=2.90, d=0.50, foot=0.090, width=0.380, pitch=0.040 |
| 4 | bounding | 0.866 | 0.739 | 0.123 | 0.000 | 0.948 | 0.121 | 1.080 | 203.616 | f=2.60, d=0.50, foot=0.090, width=0.380, pitch=-0.040 |
