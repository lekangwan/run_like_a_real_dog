# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## flat_trot_efficiency vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.898 | 0.898 | 0.116 | 0.010 | 0.920 | 0.911 | 0.739 | 1.000 | 1.000 | 0.120 |
