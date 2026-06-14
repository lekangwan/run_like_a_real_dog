# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.466, trotting=0.356, pacing=0.121, bounding=0.056

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.944 | 0.765 | 0.054 | 0.000 | 0.446 | 0.125 | 1.271 | 261.521 | f=2.60, d=0.50, foot=0.050, width=0.330, pitch=0.040 |
| 2 | trotting | 0.936 | 0.799 | 0.078 | 0.000 | 0.587 | 0.283 | 1.076 | 138.641 | f=2.60, d=0.50, foot=0.050, width=0.330, pitch=0.040 |
| 3 | pacing | 0.904 | 0.731 | 0.050 | 0.000 | 0.523 | 0.013 | 1.070 | 129.766 | f=2.50, d=0.50, foot=0.090, width=0.380, pitch=0.040 |
| 4 | bounding | 0.881 | 0.780 | 0.126 | 0.000 | 0.543 | 0.183 | 0.840 | 295.087 | f=2.60, d=0.50, foot=0.090, width=0.380, pitch=-0.040 |
