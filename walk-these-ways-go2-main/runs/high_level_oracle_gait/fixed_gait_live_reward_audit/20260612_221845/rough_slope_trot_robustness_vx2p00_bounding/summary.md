# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.706 | 0.706 | 0.848 | 0.030 | 0.373 | 0.699 | 0.491 | 1.000 | 1.000 | 0.120 |
