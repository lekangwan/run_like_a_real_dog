# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## push_lateral_pace_recovery vx=1.50

- target_gait: `pacing`
- live_best_by_weighted_metric_reward: `bounding`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bounding | 0.766 | 0.766 | 0.443 | 0.021 | 0.645 | 0.763 | 0.672 | 1.000 | 1.000 | 0.120 |
