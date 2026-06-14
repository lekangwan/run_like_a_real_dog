# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## push_lateral_pace_recovery vx=1.50

- target_gait: `pacing`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.781 | 0.781 | 0.443 | 0.022 | 0.650 | 0.776 | 0.804 | 0.500 | 1.000 | 0.080 |
