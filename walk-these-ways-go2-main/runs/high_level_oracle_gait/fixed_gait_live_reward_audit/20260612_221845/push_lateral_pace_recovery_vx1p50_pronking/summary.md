# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## push_lateral_pace_recovery vx=1.50

- target_gait: `pacing`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.773 | 0.773 | 0.537 | 0.019 | 0.572 | 0.778 | 0.735 | 0.500 | 1.000 | 0.080 |
