# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.728 | 0.728 | 0.806 | 0.028 | 0.414 | 0.724 | 0.587 | 0.500 | 1.000 | 0.080 |
