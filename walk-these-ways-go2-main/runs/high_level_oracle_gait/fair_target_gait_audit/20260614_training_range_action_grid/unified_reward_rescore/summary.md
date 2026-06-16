# Offline Unified-Reward Re-Score

This analysis re-scores the completed fair gait grid with unified, terrain-agnostic reward candidates.
Each gait is first allowed to use its own best continuous parameters under the selected candidate score.

## Candidate Summary

| candidate | top gait counts | margin counts | mean margin | mean vx_err | mean fall | mean energy | mean impact |
|---|---|---|---:|---:|---:|---:|---:|
| balanced | pacing:8 pronking:3 trotting:6 | clear_advantage:1 tie_or_noise:8 weak_advantage:8 | 0.013 | 0.264 | 0.018 | 184.9 | 0.614 |
| contact_safety | pacing:14 pronking:1 trotting:2 | clear_advantage:3 tie_or_noise:7 weak_advantage:7 | 0.019 | 0.281 | 0.019 | 182.3 | 0.501 |
| efficiency | pacing:8 pronking:1 trotting:8 | clear_advantage:2 tie_or_noise:5 weak_advantage:10 | 0.017 | 0.272 | 0.019 | 179.7 | 0.578 |
| robustness | pacing:1 pronking:9 trotting:7 | tie_or_noise:13 weak_advantage:4 | 0.007 | 0.260 | 0.017 | 194.7 | 0.769 |

## balanced

| task | vx | top | second | margin | label | score | vx_err | fall | lateral | scuff | impact | energy | params |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| flat_trot_efficiency | 0.50 | pacing | pronking | 0.023 | weak_advantage | 0.780 | 0.057 | 0.010 | 1.508 | 0.071 | 0.151 | 188.0 | f=2.50, foot=0.120, width=0.380, pitch=-0.039 |
| flat_trot_efficiency | 1.00 | pronking | trotting | 0.001 | tie_or_noise | 0.750 | 0.141 | 0.017 | 1.284 | 0.137 | 0.651 | 181.9 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| flat_trot_efficiency | 1.50 | trotting | pronking | 0.009 | tie_or_noise | 0.701 | 0.200 | 0.022 | 1.392 | 0.151 | 0.757 | 170.2 | f=3.00, foot=0.109, width=0.292, pitch=0.000 |
| flat_trot_efficiency | 2.00 | trotting | pronking | 0.023 | weak_advantage | 0.644 | 0.344 | 0.025 | 1.456 | 0.149 | 0.908 | 184.2 | f=3.38, foot=0.109, width=0.292, pitch=0.000 |
| push_lateral_pace_recovery | 1.20 | trotting | pronking | 0.016 | weak_advantage | 0.731 | 0.169 | 0.016 | 1.347 | 0.165 | 0.653 | 147.9 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.50 | trotting | pronking | 0.012 | weak_advantage | 0.707 | 0.202 | 0.017 | 1.266 | 0.146 | 0.717 | 156.8 | f=3.00, foot=0.109, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.80 | trotting | pacing | 0.016 | weak_advantage | 0.668 | 0.315 | 0.021 | 1.403 | 0.166 | 0.704 | 162.5 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 0.50 | pacing | pronking | 0.016 | weak_advantage | 0.804 | 0.093 | 0.008 | 1.226 | 0.015 | 0.176 | 223.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.00 | pacing | pronking | 0.009 | tie_or_noise | 0.758 | 0.150 | 0.016 | 1.521 | 0.041 | 0.470 | 174.5 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.50 | pronking | pacing | 0.008 | tie_or_noise | 0.703 | 0.244 | 0.022 | 1.583 | 0.054 | 0.855 | 181.8 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 2.00 | pronking | trotting | 0.008 | tie_or_noise | 0.619 | 0.421 | 0.023 | 1.504 | 0.056 | 0.927 | 198.5 | f=3.00, foot=0.109, width=0.330, pitch=0.038 |
| rough_slope_trot_robustness | 0.50 | pacing | trotting | 0.006 | tie_or_noise | 0.772 | 0.098 | 0.008 | 1.773 | 0.024 | 0.276 | 200.6 | f=2.50, foot=0.120, width=0.340, pitch=0.040 |
| rough_slope_trot_robustness | 1.00 | pacing | pronking | 0.007 | tie_or_noise | 0.733 | 0.203 | 0.018 | 1.307 | 0.040 | 0.457 | 163.8 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 1.50 | pacing | pronking | 0.006 | tie_or_noise | 0.654 | 0.293 | 0.022 | 1.336 | 0.058 | 0.742 | 179.3 | f=2.89, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 2.00 | trotting | pacing | 0.012 | weak_advantage | 0.553 | 0.493 | 0.024 | 1.352 | 0.072 | 0.708 | 189.2 | f=2.62, foot=0.109, width=0.330, pitch=0.038 |
| stepping_stones_easy_bound_highspeed | 1.70 | pacing | pronking | 0.010 | weak_advantage | 0.550 | 0.482 | 0.018 | 1.695 | 0.115 | 0.712 | 230.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | pacing | bounding | 0.035 | clear_advantage | 0.530 | 0.587 | 0.025 | 1.770 | 0.102 | 0.569 | 211.3 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |

## contact_safety

| task | vx | top | second | margin | label | score | vx_err | fall | lateral | scuff | impact | energy | params |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| flat_trot_efficiency | 0.50 | pacing | trotting | 0.063 | clear_advantage | 0.790 | 0.057 | 0.010 | 1.508 | 0.071 | 0.151 | 188.0 | f=2.50, foot=0.120, width=0.380, pitch=-0.039 |
| flat_trot_efficiency | 1.00 | pacing | trotting | 0.030 | clear_advantage | 0.747 | 0.136 | 0.018 | 1.544 | 0.068 | 0.318 | 146.1 | f=2.50, foot=0.120, width=0.341, pitch=-0.039 |
| flat_trot_efficiency | 1.50 | pacing | trotting | 0.010 | tie_or_noise | 0.689 | 0.258 | 0.019 | 1.447 | 0.125 | 0.424 | 167.4 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |
| flat_trot_efficiency | 2.00 | pacing | trotting | 0.005 | tie_or_noise | 0.644 | 0.439 | 0.028 | 1.439 | 0.099 | 0.524 | 179.1 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| push_lateral_pace_recovery | 1.20 | pacing | trotting | 0.006 | tie_or_noise | 0.711 | 0.190 | 0.022 | 1.580 | 0.087 | 0.480 | 145.1 | f=2.50, foot=0.120, width=0.341, pitch=0.039 |
| push_lateral_pace_recovery | 1.50 | trotting | pacing | 0.000 | tie_or_noise | 0.688 | 0.202 | 0.017 | 1.266 | 0.146 | 0.717 | 156.8 | f=3.00, foot=0.109, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.80 | trotting | pacing | 0.001 | tie_or_noise | 0.656 | 0.298 | 0.022 | 1.211 | 0.151 | 0.775 | 173.0 | f=3.00, foot=0.109, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 0.50 | pacing | trotting | 0.062 | clear_advantage | 0.824 | 0.093 | 0.008 | 1.226 | 0.015 | 0.176 | 223.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.00 | pacing | trotting | 0.023 | weak_advantage | 0.763 | 0.150 | 0.016 | 1.521 | 0.041 | 0.470 | 174.5 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.50 | pacing | trotting | 0.018 | weak_advantage | 0.724 | 0.288 | 0.022 | 1.376 | 0.027 | 0.446 | 170.2 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |
| ramp_up_trot_robustness | 2.00 | pronking | pacing | 0.001 | tie_or_noise | 0.641 | 0.421 | 0.023 | 1.504 | 0.056 | 0.927 | 198.5 | f=3.00, foot=0.109, width=0.330, pitch=0.038 |
| rough_slope_trot_robustness | 0.50 | pacing | trotting | 0.027 | weak_advantage | 0.789 | 0.098 | 0.008 | 1.773 | 0.024 | 0.276 | 200.6 | f=2.50, foot=0.120, width=0.340, pitch=0.040 |
| rough_slope_trot_robustness | 1.00 | pacing | trotting | 0.021 | weak_advantage | 0.747 | 0.203 | 0.018 | 1.307 | 0.040 | 0.457 | 163.8 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 1.50 | pacing | pronking | 0.015 | weak_advantage | 0.676 | 0.313 | 0.023 | 1.604 | 0.043 | 0.466 | 162.4 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 2.00 | pacing | trotting | 0.013 | weak_advantage | 0.591 | 0.563 | 0.023 | 1.266 | 0.051 | 0.633 | 208.8 | f=2.50, foot=0.120, width=0.342, pitch=0.038 |
| stepping_stones_easy_bound_highspeed | 1.70 | pacing | bounding | 0.004 | tie_or_noise | 0.578 | 0.482 | 0.018 | 1.695 | 0.115 | 0.712 | 230.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | pacing | bounding | 0.027 | weak_advantage | 0.577 | 0.587 | 0.025 | 1.770 | 0.102 | 0.569 | 211.3 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |

## efficiency

| task | vx | top | second | margin | label | score | vx_err | fall | lateral | scuff | impact | energy | params |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| flat_trot_efficiency | 0.50 | pacing | trotting | 0.019 | weak_advantage | 0.795 | 0.057 | 0.010 | 1.508 | 0.071 | 0.151 | 188.0 | f=2.50, foot=0.120, width=0.380, pitch=-0.039 |
| flat_trot_efficiency | 1.00 | trotting | pronking | 0.006 | tie_or_noise | 0.767 | 0.129 | 0.018 | 1.069 | 0.162 | 0.613 | 145.2 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| flat_trot_efficiency | 1.50 | trotting | pacing | 0.018 | weak_advantage | 0.707 | 0.210 | 0.018 | 1.517 | 0.172 | 0.576 | 152.0 | f=3.00, foot=0.080, width=0.330, pitch=0.038 |
| flat_trot_efficiency | 2.00 | trotting | pronking | 0.026 | weak_advantage | 0.630 | 0.395 | 0.028 | 1.290 | 0.174 | 0.858 | 171.5 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.20 | trotting | pronking | 0.026 | weak_advantage | 0.749 | 0.169 | 0.016 | 1.347 | 0.165 | 0.653 | 147.9 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.50 | trotting | pronking | 0.020 | weak_advantage | 0.710 | 0.240 | 0.020 | 1.609 | 0.170 | 0.678 | 159.0 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.80 | trotting | pronking | 0.033 | clear_advantage | 0.675 | 0.315 | 0.021 | 1.403 | 0.166 | 0.704 | 162.5 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 0.50 | pacing | trotting | 0.016 | weak_advantage | 0.808 | 0.093 | 0.008 | 1.226 | 0.015 | 0.176 | 223.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.00 | pacing | trotting | 0.012 | weak_advantage | 0.761 | 0.150 | 0.016 | 1.521 | 0.041 | 0.470 | 174.5 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| ramp_up_trot_robustness | 1.50 | pronking | pacing | 0.014 | weak_advantage | 0.695 | 0.244 | 0.022 | 1.583 | 0.054 | 0.855 | 181.8 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 2.00 | trotting | pronking | 0.004 | tie_or_noise | 0.583 | 0.449 | 0.024 | 1.419 | 0.044 | 0.910 | 192.3 | f=3.38, foot=0.080, width=0.330, pitch=0.000 |
| rough_slope_trot_robustness | 0.50 | pacing | trotting | 0.000 | tie_or_noise | 0.779 | 0.098 | 0.008 | 1.773 | 0.024 | 0.276 | 200.6 | f=2.50, foot=0.120, width=0.340, pitch=0.040 |
| rough_slope_trot_robustness | 1.00 | pacing | pronking | 0.007 | tie_or_noise | 0.739 | 0.203 | 0.018 | 1.307 | 0.040 | 0.457 | 163.8 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 1.50 | pacing | trotting | 0.010 | tie_or_noise | 0.646 | 0.313 | 0.023 | 1.604 | 0.043 | 0.466 | 162.4 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| rough_slope_trot_robustness | 2.00 | trotting | pacing | 0.014 | weak_advantage | 0.532 | 0.493 | 0.024 | 1.352 | 0.072 | 0.708 | 189.2 | f=2.62, foot=0.109, width=0.330, pitch=0.038 |
| stepping_stones_easy_bound_highspeed | 1.70 | pacing | pronking | 0.019 | weak_advantage | 0.541 | 0.482 | 0.018 | 1.695 | 0.115 | 0.712 | 230.0 | f=2.50, foot=0.120, width=0.380, pitch=0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | pacing | trotting | 0.040 | clear_advantage | 0.511 | 0.587 | 0.025 | 1.770 | 0.102 | 0.569 | 211.3 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |

## robustness

| task | vx | top | second | margin | label | score | vx_err | fall | lateral | scuff | impact | energy | params |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| flat_trot_efficiency | 0.50 | pronking | trotting | 0.005 | tie_or_noise | 0.759 | 0.059 | 0.006 | 0.697 | 0.129 | 0.820 | 250.6 | f=3.39, foot=0.080, width=0.330, pitch=0.000 |
| flat_trot_efficiency | 1.00 | pronking | trotting | 0.001 | tie_or_noise | 0.751 | 0.141 | 0.017 | 1.284 | 0.137 | 0.651 | 181.9 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| flat_trot_efficiency | 1.50 | trotting | pronking | 0.001 | tie_or_noise | 0.705 | 0.200 | 0.022 | 1.392 | 0.151 | 0.757 | 170.2 | f=3.00, foot=0.109, width=0.292, pitch=0.000 |
| flat_trot_efficiency | 2.00 | trotting | pronking | 0.008 | tie_or_noise | 0.663 | 0.344 | 0.025 | 1.456 | 0.149 | 0.908 | 184.2 | f=3.38, foot=0.109, width=0.292, pitch=0.000 |
| push_lateral_pace_recovery | 1.20 | trotting | pronking | 0.008 | tie_or_noise | 0.729 | 0.169 | 0.016 | 1.347 | 0.165 | 0.653 | 147.9 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.50 | trotting | pronking | 0.005 | tie_or_noise | 0.711 | 0.202 | 0.017 | 1.266 | 0.146 | 0.717 | 156.8 | f=3.00, foot=0.109, width=0.330, pitch=0.000 |
| push_lateral_pace_recovery | 1.80 | trotting | pronking | 0.008 | tie_or_noise | 0.679 | 0.298 | 0.022 | 1.211 | 0.151 | 0.775 | 173.0 | f=3.00, foot=0.109, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 0.50 | pronking | pacing | 0.002 | tie_or_noise | 0.781 | 0.070 | 0.007 | 1.579 | 0.044 | 0.845 | 256.5 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 1.00 | pronking | pacing | 0.009 | tie_or_noise | 0.745 | 0.143 | 0.014 | 1.152 | 0.056 | 0.723 | 196.1 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 1.50 | pronking | trotting | 0.015 | weak_advantage | 0.704 | 0.244 | 0.022 | 1.583 | 0.054 | 0.855 | 181.8 | f=3.00, foot=0.080, width=0.330, pitch=0.000 |
| ramp_up_trot_robustness | 2.00 | pronking | trotting | 0.013 | weak_advantage | 0.636 | 0.421 | 0.023 | 1.504 | 0.056 | 0.927 | 198.5 | f=3.00, foot=0.109, width=0.330, pitch=0.038 |
| rough_slope_trot_robustness | 0.50 | trotting | pronking | 0.001 | tie_or_noise | 0.755 | 0.092 | 0.006 | 1.148 | 0.061 | 0.565 | 224.7 | f=3.00, foot=0.110, width=0.291, pitch=0.039 |
| rough_slope_trot_robustness | 1.00 | pronking | trotting | 0.008 | tie_or_noise | 0.720 | 0.167 | 0.012 | 1.536 | 0.060 | 0.922 | 189.8 | f=3.00, foot=0.109, width=0.291, pitch=0.000 |
| rough_slope_trot_robustness | 1.50 | pronking | pacing | 0.002 | tie_or_noise | 0.647 | 0.313 | 0.020 | 1.539 | 0.073 | 0.722 | 173.8 | f=3.39, foot=0.109, width=0.291, pitch=0.039 |
| rough_slope_trot_robustness | 2.00 | trotting | pacing | 0.012 | weak_advantage | 0.565 | 0.493 | 0.024 | 1.352 | 0.072 | 0.708 | 189.2 | f=2.62, foot=0.109, width=0.330, pitch=0.038 |
| stepping_stones_easy_bound_highspeed | 1.70 | pronking | pacing | 0.003 | tie_or_noise | 0.554 | 0.470 | 0.019 | 1.608 | 0.117 | 0.949 | 223.4 | f=3.39, foot=0.109, width=0.369, pitch=0.000 |
| stepping_stones_easy_bound_highspeed | 2.00 | pacing | trotting | 0.024 | weak_advantage | 0.541 | 0.587 | 0.025 | 1.770 | 0.102 | 0.569 | 211.3 | f=2.50, foot=0.120, width=0.380, pitch=0.038 |