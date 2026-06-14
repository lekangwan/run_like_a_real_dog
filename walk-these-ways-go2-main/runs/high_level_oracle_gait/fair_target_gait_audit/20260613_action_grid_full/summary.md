# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.427, trotting=0.395, pacing=0.132, bounding=0.046

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.917 | 0.723 | 0.066 | 0.007 | 1.990 | 0.121 | 1.463 | 252.866 | f=2.61, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.915 | 0.782 | 0.056 | 0.007 | 1.154 | 0.205 | 0.912 | 130.257 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.882 | 0.713 | 0.058 | 0.006 | 1.282 | 0.140 | 0.976 | 195.617 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | bounding | 0.851 | 0.746 | 0.142 | 0.005 | 1.356 | 0.147 | 0.984 | 239.833 | f=2.60, d=0.50, foot=0.090, width=0.380, pitch=0.000 |

## flat_trot_efficiency vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.478, trotting=0.369, pacing=0.114, bounding=0.038

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.694 | 0.135 | 0.013 | 1.606 | 0.115 | 1.651 | 199.576 | f=2.61, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 2 | trotting | 0.884 | 0.746 | 0.122 | 0.016 | 1.597 | 0.210 | 1.168 | 126.504 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.849 | 0.710 | 0.111 | 0.012 | 1.553 | 0.148 | 1.363 | 162.929 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.000 |
| 4 | bounding | 0.816 | 0.709 | 0.150 | 0.017 | 1.669 | 0.171 | 1.311 | 191.860 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |

## flat_trot_efficiency vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.482, pronking=0.329, pacing=0.125, bounding=0.064

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.828 | 0.723 | 0.202 | 0.016 | 1.550 | 0.150 | 1.340 | 162.839 | f=3.00, d=0.50, foot=0.109, width=0.292, pitch=0.038 |
| 2 | pronking | 0.816 | 0.700 | 0.245 | 0.023 | 1.587 | 0.153 | 1.535 | 181.690 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 3 | pacing | 0.787 | 0.671 | 0.229 | 0.021 | 1.561 | 0.119 | 1.317 | 169.859 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.768 | 0.710 | 0.236 | 0.022 | 1.401 | 0.145 | 1.571 | 196.124 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.000 |

## flat_trot_efficiency vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.688, pronking=0.139, pacing=0.109, bounding=0.063

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.739 | 0.707 | 0.339 | 0.023 | 1.382 | 0.151 | 1.487 | 181.087 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | pronking | 0.692 | 0.796 | 0.452 | 0.019 | 1.366 | 0.169 | 1.647 | 180.636 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.000 |
| 3 | pacing | 0.684 | 0.671 | 0.382 | 0.026 | 1.390 | 0.127 | 1.543 | 186.463 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.668 | 0.657 | 0.392 | 0.026 | 1.572 | 0.139 | 1.670 | 203.551 | f=3.38, d=0.50, foot=0.091, width=0.380, pitch=0.000 |

## push_lateral_pace_recovery vx=1.50

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.465, trotting=0.330, bounding=0.105, pacing=0.100

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.701 | 0.752 | 0.274 | 0.017 | 1.217 | 0.151 | 1.399 | 165.605 | f=3.00, d=0.50, foot=0.051, width=0.291, pitch=0.000 |
| 2 | trotting | 0.691 | 0.717 | 0.236 | 0.020 | 1.286 | 0.203 | 1.203 | 148.354 | f=3.38, d=0.50, foot=0.051, width=0.292, pitch=0.038 |
| 3 | bounding | 0.656 | 0.759 | 0.256 | 0.020 | 1.300 | 0.143 | 1.496 | 198.216 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | pacing | 0.655 | 0.684 | 0.230 | 0.015 | 1.316 | 0.134 | 1.405 | 170.643 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.038 |

## ramp_up_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.473, trotting=0.342, pacing=0.146, bounding=0.039

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.827 | 0.071 | 0.007 | 1.582 | 0.035 | 1.450 | 252.716 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.882 | 0.746 | 0.068 | 0.007 | 1.817 | 0.069 | 0.896 | 162.567 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.856 | 0.727 | 0.097 | 0.009 | 1.330 | 0.046 | 0.776 | 248.679 | f=2.50, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.817 | 0.697 | 0.109 | 0.006 | 1.640 | 0.077 | 1.057 | 239.962 | f=3.40, d=0.50, foot=0.090, width=0.380, pitch=0.040 |

## ramp_up_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.533, trotting=0.318, pacing=0.123, bounding=0.026

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.863 | 0.747 | 0.126 | 0.013 | 1.310 | 0.064 | 1.265 | 161.539 | f=3.39, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 2 | trotting | 0.848 | 0.733 | 0.145 | 0.014 | 1.521 | 0.060 | 1.097 | 146.929 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.820 | 0.686 | 0.145 | 0.017 | 1.592 | 0.050 | 1.194 | 163.096 | f=2.89, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
| 4 | bounding | 0.773 | 0.708 | 0.168 | 0.012 | 1.673 | 0.059 | 1.195 | 187.050 | f=3.39, d=0.50, foot=0.091, width=0.380, pitch=0.000 |

## ramp_up_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.690, trotting=0.220, pacing=0.069, bounding=0.020

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.795 | 0.727 | 0.226 | 0.022 | 1.410 | 0.054 | 1.627 | 180.267 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=0.038 |
| 2 | trotting | 0.761 | 0.703 | 0.258 | 0.020 | 1.527 | 0.053 | 1.323 | 160.931 | f=3.39, d=0.50, foot=0.051, width=0.369, pitch=0.039 |
| 3 | pacing | 0.726 | 0.676 | 0.271 | 0.018 | 1.453 | 0.030 | 1.472 | 186.500 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | bounding | 0.689 | 0.666 | 0.279 | 0.019 | 1.454 | 0.048 | 1.416 | 194.021 | f=3.39, d=0.50, foot=0.091, width=0.341, pitch=0.039 |

## ramp_up_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.531, trotting=0.376, pacing=0.072, bounding=0.020

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.644 | 0.700 | 0.445 | 0.023 | 1.247 | 0.060 | 1.890 | 201.450 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | trotting | 0.633 | 0.688 | 0.418 | 0.020 | 1.243 | 0.040 | 1.575 | 184.113 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=-0.038 |
| 3 | pacing | 0.584 | 0.669 | 0.499 | 0.027 | 1.226 | 0.035 | 1.608 | 207.399 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.546 | 0.651 | 0.487 | 0.023 | 1.358 | 0.039 | 1.812 | 236.272 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.038 |

## rough_slope_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.521, trotting=0.335, pacing=0.115, bounding=0.029

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.842 | 0.749 | 0.093 | 0.006 | 1.440 | 0.061 | 1.097 | 237.564 | f=3.00, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 2 | trotting | 0.829 | 0.741 | 0.090 | 0.006 | 1.770 | 0.083 | 0.893 | 180.503 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.797 | 0.676 | 0.098 | 0.007 | 1.909 | 0.044 | 0.974 | 193.635 | f=2.50, d=0.50, foot=0.120, width=0.340, pitch=0.040 |
| 4 | bounding | 0.756 | 0.674 | 0.129 | 0.008 | 1.247 | 0.063 | 1.174 | 253.928 | f=3.39, d=0.50, foot=0.120, width=0.341, pitch=0.039 |

## rough_slope_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.527, trotting=0.328, pacing=0.103, bounding=0.042

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.783 | 0.725 | 0.166 | 0.015 | 1.424 | 0.065 | 1.555 | 186.730 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 2 | trotting | 0.769 | 0.696 | 0.163 | 0.017 | 1.746 | 0.064 | 1.291 | 169.216 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.735 | 0.661 | 0.178 | 0.016 | 1.361 | 0.071 | 1.150 | 147.794 | f=2.50, d=0.50, foot=0.091, width=0.341, pitch=0.039 |
| 4 | bounding | 0.707 | 0.672 | 0.180 | 0.011 | 1.842 | 0.057 | 1.311 | 189.429 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |

## rough_slope_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.449, pronking=0.274, pacing=0.232, bounding=0.045

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.669 | 0.694 | 0.293 | 0.019 | 1.498 | 0.064 | 1.367 | 171.494 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 2 | pronking | 0.654 | 0.689 | 0.319 | 0.018 | 1.384 | 0.068 | 1.567 | 191.382 | f=3.39, d=0.50, foot=0.109, width=0.330, pitch=0.039 |
| 3 | pacing | 0.649 | 0.646 | 0.304 | 0.022 | 1.528 | 0.044 | 1.484 | 181.591 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.600 | 0.652 | 0.349 | 0.022 | 1.398 | 0.061 | 1.583 | 209.680 | f=3.38, d=0.50, foot=0.120, width=0.380, pitch=0.000 |

## rough_slope_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.440, trotting=0.433, pronking=0.089, bounding=0.038

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.530 | 0.607 | 0.495 | 0.028 | 1.383 | 0.057 | 1.689 | 204.517 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.529 | 0.628 | 0.491 | 0.023 | 1.439 | 0.057 | 1.506 | 200.898 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=-0.038 |
| 3 | pronking | 0.481 | 0.654 | 0.577 | 0.022 | 1.592 | 0.067 | 1.900 | 221.815 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 4 | bounding | 0.456 | 0.597 | 0.608 | 0.024 | 1.383 | 0.065 | 1.726 | 240.791 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.038 |

## stepping_stones_easy_bound_highspeed vx=2.00

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.442, trotting=0.319, bounding=0.139, pronking=0.099

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.549 | 0.681 | 0.587 | 0.022 | 1.390 | 0.114 | 1.556 | 220.003 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.539 | 0.665 | 0.635 | 0.019 | 1.319 | 0.127 | 1.548 | 234.228 | f=3.00, d=0.50, foot=0.109, width=0.368, pitch=0.038 |
| 3 | bounding | 0.514 | 0.644 | 0.709 | 0.020 | 1.605 | 0.105 | 1.452 | 269.422 | f=2.62, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | pronking | 0.504 | 0.671 | 0.720 | 0.024 | 1.368 | 0.127 | 1.561 | 266.343 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
