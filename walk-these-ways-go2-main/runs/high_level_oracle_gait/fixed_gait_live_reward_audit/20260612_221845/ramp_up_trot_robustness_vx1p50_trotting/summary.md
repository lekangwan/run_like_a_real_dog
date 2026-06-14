# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## ramp_up_trot_robustness vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.854 | 0.854 | 0.285 | 0.024 | 0.772 | 0.802 | 0.836 | 0.500 | 1.000 | 0.080 |
