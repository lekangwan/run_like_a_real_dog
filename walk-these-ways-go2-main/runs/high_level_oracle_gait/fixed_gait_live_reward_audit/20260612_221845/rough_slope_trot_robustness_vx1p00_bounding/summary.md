# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.805 | 0.805 | 0.225 | 0.020 | 0.793 | 0.827 | 0.658 | 1.000 | 1.000 | 0.120 |
