# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.428, trotting=0.399, pacing=0.133, bounding=0.041

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.889 | 0.709 | 0.139 | 0.014 | 1.606 | 0.143 | 1.201 | 162.835 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 2 | trotting | 0.887 | 0.748 | 0.122 | 0.011 | 1.486 | 0.204 | 1.073 | 124.034 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.854 | 0.703 | 0.118 | 0.014 | 1.330 | 0.149 | 1.293 | 165.978 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.819 | 0.702 | 0.154 | 0.013 | 1.638 | 0.200 | 1.103 | 167.411 | f=3.39, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
