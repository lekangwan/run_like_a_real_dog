# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## ramp_up_trot_robustness vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.823 | 0.823 | 0.496 | 0.030 | 0.652 | 0.745 | 0.806 | 0.500 | 1.000 | 0.080 |
