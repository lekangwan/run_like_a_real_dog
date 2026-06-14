# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## flat_trot_efficiency vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.876 | 0.876 | 0.168 | 0.019 | 0.869 | 0.854 | 0.764 | 1.000 | 1.000 | 0.120 |
