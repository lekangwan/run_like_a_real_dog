# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pacing`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pacing | 0.802 | 0.802 | 0.184 | 0.019 | 0.797 | 0.830 | 0.803 | 1.000 | 1.000 | 0.120 |
