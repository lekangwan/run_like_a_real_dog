# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## flat_trot_efficiency vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.839 | 0.839 | 0.135 | 0.018 | 0.892 | 0.876 | 0.900 | 0.500 | 1.000 | 0.080 |
