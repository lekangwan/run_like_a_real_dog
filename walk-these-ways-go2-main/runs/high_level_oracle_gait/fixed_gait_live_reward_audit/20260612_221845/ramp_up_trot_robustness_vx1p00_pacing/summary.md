# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## ramp_up_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pacing`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pacing | 0.860 | 0.860 | 0.163 | 0.019 | 0.815 | 0.833 | 0.873 | 1.000 | 1.000 | 0.120 |
