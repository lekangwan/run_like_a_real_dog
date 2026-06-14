# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.850 | 0.850 | 0.092 | 0.009 | 0.933 | 0.928 | 0.781 | 0.500 | 1.000 | 0.080 |
