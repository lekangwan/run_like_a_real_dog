# Unified Reward Candidate Selection

Offline re-score source: `../fair_gait_grid_results.csv`.
Each candidate is a single terrain-agnostic weighting over physical scores; no task-specific weights or gait priors are used.

| candidate | decision | rationale |
|---|---|---|
| balanced | secondary candidate | Middle-ground trade-off with more diversity than contact_safety and less weak signal than robustness. Some low-speed flat/ramp pacing choices need live-audit scrutiny. |
| contact_safety | reject as mainline | Overweights contact/scuff/impact enough that it collapses to pacing in 14/17 task-speed rows, including many flat/ramp/rough speeds. Useful as an ablation, not a main reward. |
| efficiency | primary candidate | Best overall energy among the candidates, relatively low impact, useful margins, and does not collapse completely to one gait. It favors trot for flat mid/high and push, and pace for contact-heavy ramp/rough/stones cases. |
| robustness | diagnostic only | Best mean vx/fall numbers, but margins are mostly tie/noise and impact/energy are worst. Likely weak PPO signal and may still rely on high-impact pronk. |

## Candidate Stats

| candidate | top gait counts | margin counts | mean margin | mean vx_err | mean fall | mean energy | mean impact |
|---|---|---|---:|---:|---:|---:|---:|
| balanced | pacing:8 pronking:3 trotting:6 | clear_advantage:1 tie_or_noise:8 weak_advantage:8 | 0.013 | 0.264 | 0.018 | 184.9 | 0.614 |
| contact_safety | pacing:14 pronking:1 trotting:2 | clear_advantage:3 tie_or_noise:7 weak_advantage:7 | 0.019 | 0.281 | 0.019 | 182.3 | 0.501 |
| efficiency | pacing:8 pronking:1 trotting:8 | clear_advantage:2 tie_or_noise:5 weak_advantage:10 | 0.017 | 0.272 | 0.019 | 179.7 | 0.578 |
| robustness | pacing:1 pronking:9 trotting:7 | tie_or_noise:13 weak_advantage:4 | 0.007 | 0.260 | 0.017 | 194.7 | 0.769 |

## Next Step

Implement or emulate `efficiency` first in the live wrapper and run a fixed/fair reward audit to verify that the live reward ranking matches the offline re-score. Keep `balanced` as the second candidate if efficiency looks too biased toward pace/contact-heavy behavior.