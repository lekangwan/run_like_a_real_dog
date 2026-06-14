# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## rough_slope_trot_robustness vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.786 | 0.786 | 0.358 | 0.023 | 0.680 | 0.761 | 0.730 | 0.500 | 1.000 | 0.080 |
