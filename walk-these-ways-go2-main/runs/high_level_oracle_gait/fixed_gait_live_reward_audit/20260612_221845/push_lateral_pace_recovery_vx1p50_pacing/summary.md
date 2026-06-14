# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## push_lateral_pace_recovery vx=1.50

- target_gait: `pacing`
- live_best_by_weighted_metric_reward: `pacing`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pacing | 0.758 | 0.758 | 0.457 | 0.022 | 0.603 | 0.716 | 0.801 | 1.000 | 1.000 | 0.120 |
