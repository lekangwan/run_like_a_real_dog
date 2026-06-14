# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## ramp_up_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.887 | 0.887 | 0.137 | 0.018 | 0.887 | 0.866 | 0.870 | 0.500 | 1.000 | 0.080 |
