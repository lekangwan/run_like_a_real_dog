# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.478, trotting=0.369, pacing=0.114, bounding=0.038

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.694 | 0.135 | 0.013 | 1.606 | 0.115 | 1.651 | 199.576 | f=2.61, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 2 | trotting | 0.884 | 0.746 | 0.122 | 0.016 | 1.597 | 0.210 | 1.168 | 126.504 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.849 | 0.710 | 0.111 | 0.012 | 1.553 | 0.148 | 1.363 | 162.929 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.000 |
| 4 | bounding | 0.816 | 0.709 | 0.150 | 0.017 | 1.669 | 0.171 | 1.311 | 191.860 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
