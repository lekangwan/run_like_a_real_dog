# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.531, trotting=0.376, pacing=0.072, bounding=0.020

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.644 | 0.700 | 0.445 | 0.023 | 1.247 | 0.060 | 1.890 | 201.450 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | trotting | 0.633 | 0.688 | 0.418 | 0.020 | 1.243 | 0.040 | 1.575 | 184.113 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=-0.038 |
| 3 | pacing | 0.584 | 0.669 | 0.499 | 0.027 | 1.226 | 0.035 | 1.608 | 207.399 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.546 | 0.651 | 0.487 | 0.023 | 1.358 | 0.039 | 1.812 | 236.272 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
