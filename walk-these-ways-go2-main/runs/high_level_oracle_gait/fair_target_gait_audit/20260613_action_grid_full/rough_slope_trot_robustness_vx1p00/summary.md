# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## rough_slope_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.527, trotting=0.328, pacing=0.103, bounding=0.042

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.783 | 0.725 | 0.166 | 0.015 | 1.424 | 0.065 | 1.555 | 186.730 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 2 | trotting | 0.769 | 0.696 | 0.163 | 0.017 | 1.746 | 0.064 | 1.291 | 169.216 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.735 | 0.661 | 0.178 | 0.016 | 1.361 | 0.071 | 1.150 | 147.794 | f=2.50, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
| 4 | bounding | 0.707 | 0.672 | 0.180 | 0.011 | 1.842 | 0.057 | 1.311 | 189.429 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
