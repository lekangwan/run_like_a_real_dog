# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pacing`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pacing | 0.820 | 0.820 | 0.111 | 0.008 | 0.873 | 0.899 | 0.832 | 1.000 | 1.000 | 0.120 |
