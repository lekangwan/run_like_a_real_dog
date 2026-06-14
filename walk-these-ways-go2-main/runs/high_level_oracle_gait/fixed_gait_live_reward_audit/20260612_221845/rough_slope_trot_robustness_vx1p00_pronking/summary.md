# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.833 | 0.833 | 0.200 | 0.018 | 0.824 | 0.858 | 0.763 | 0.500 | 1.000 | 0.080 |
