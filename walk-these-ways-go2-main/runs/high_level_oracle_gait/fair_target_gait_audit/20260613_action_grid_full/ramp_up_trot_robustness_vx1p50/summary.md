# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.690, trotting=0.220, pacing=0.069, bounding=0.020

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.795 | 0.727 | 0.226 | 0.022 | 1.410 | 0.054 | 1.627 | 180.267 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=0.038 |
| 2 | trotting | 0.761 | 0.703 | 0.258 | 0.020 | 1.527 | 0.053 | 1.323 | 160.931 | f=3.39, d=0.50, foot=0.051, width=0.369, pitch=0.039 |
| 3 | pacing | 0.726 | 0.676 | 0.271 | 0.018 | 1.453 | 0.030 | 1.472 | 186.500 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | bounding | 0.689 | 0.666 | 0.279 | 0.019 | 1.454 | 0.048 | 1.416 | 194.021 | f=3.39, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
