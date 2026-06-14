# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## flat_trot_efficiency vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.846 | 0.846 | 0.264 | 0.026 | 0.791 | 0.783 | 0.764 | 1.000 | 1.000 | 0.120 |
