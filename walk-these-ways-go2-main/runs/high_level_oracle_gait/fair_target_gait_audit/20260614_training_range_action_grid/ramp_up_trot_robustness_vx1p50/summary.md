# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## ramp_up_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.522, trotting=0.335, pacing=0.115, bounding=0.028

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.776 | 0.723 | 0.248 | 0.021 | 1.686 | 0.047 | 1.719 | 189.427 | f=3.00, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 2 | trotting | 0.763 | 0.709 | 0.246 | 0.019 | 1.633 | 0.046 | 1.217 | 164.122 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 3 | pacing | 0.731 | 0.728 | 0.288 | 0.022 | 1.376 | 0.027 | 1.243 | 170.224 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.689 | 0.687 | 0.285 | 0.022 | 1.585 | 0.034 | 1.419 | 188.284 | f=3.00, d=0.50, foot=0.091, width=0.380, pitch=0.000 |
