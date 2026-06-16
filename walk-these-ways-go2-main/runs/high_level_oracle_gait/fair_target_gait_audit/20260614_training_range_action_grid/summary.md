# Fair Target-Gait Audit

This audit gives every gait an equal continuous-parameter search budget.
It reports neutral weighted scores, raw metrics, and Pareto candidates.

## flat_trot_efficiency vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.445, trotting=0.374, pacing=0.131, bounding=0.050

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.919 | 0.722 | 0.072 | 0.010 | 1.212 | 0.115 | 1.197 | 246.220 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.914 | 0.766 | 0.065 | 0.008 | 1.593 | 0.199 | 0.928 | 133.956 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.883 | 0.720 | 0.057 | 0.008 | 1.155 | 0.152 | 1.212 | 221.696 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.854 | 0.731 | 0.149 | 0.006 | 1.651 | 0.159 | 1.052 | 232.432 | f=2.61, d=0.50, foot=0.090, width=0.341, pitch=0.039 |

## flat_trot_efficiency vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.428, trotting=0.399, pacing=0.133, bounding=0.041

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.889 | 0.709 | 0.139 | 0.014 | 1.606 | 0.143 | 1.201 | 162.835 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 2 | trotting | 0.887 | 0.748 | 0.122 | 0.011 | 1.486 | 0.204 | 1.073 | 124.034 | f=2.61, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | pacing | 0.854 | 0.703 | 0.118 | 0.014 | 1.330 | 0.149 | 1.293 | 165.978 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.819 | 0.702 | 0.154 | 0.013 | 1.638 | 0.200 | 1.103 | 167.411 | f=3.39, d=0.50, foot=0.091, width=0.341, pitch=0.039 |

## flat_trot_efficiency vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.475, pronking=0.369, pacing=0.112, bounding=0.044

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.828 | 0.741 | 0.205 | 0.023 | 1.342 | 0.161 | 1.470 | 165.609 | f=2.61, d=0.50, foot=0.109, width=0.330, pitch=0.000 |
| 2 | pronking | 0.820 | 0.682 | 0.245 | 0.018 | 1.653 | 0.141 | 1.640 | 180.742 | f=3.00, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 3 | pacing | 0.784 | 0.696 | 0.240 | 0.022 | 1.589 | 0.130 | 1.378 | 179.967 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.756 | 0.705 | 0.243 | 0.018 | 1.626 | 0.154 | 1.499 | 191.935 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=0.000 |

## flat_trot_efficiency vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.733, pronking=0.115, pacing=0.103, bounding=0.049

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.743 | 0.705 | 0.344 | 0.025 | 1.456 | 0.149 | 1.511 | 184.199 | f=3.38, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| 2 | pronking | 0.688 | 0.671 | 0.455 | 0.026 | 1.467 | 0.147 | 1.603 | 190.859 | f=3.38, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 3 | pacing | 0.685 | 0.649 | 0.388 | 0.023 | 1.216 | 0.126 | 1.527 | 189.524 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=-0.038 |
| 4 | bounding | 0.663 | 0.648 | 0.390 | 0.027 | 1.419 | 0.145 | 1.799 | 214.598 | f=3.38, d=0.50, foot=0.120, width=0.342, pitch=-0.038 |

## push_lateral_pace_recovery vx=1.20

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.528, trotting=0.284, bounding=0.121, pacing=0.067

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.733 | 0.747 | 0.170 | 0.016 | 1.205 | 0.128 | 1.616 | 183.665 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.000 |
| 2 | trotting | 0.714 | 0.744 | 0.162 | 0.018 | 1.034 | 0.199 | 1.101 | 140.352 | f=3.39, d=0.50, foot=0.051, width=0.291, pitch=0.039 |
| 3 | bounding | 0.689 | 0.723 | 0.191 | 0.019 | 1.137 | 0.159 | 1.401 | 191.320 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 4 | pacing | 0.671 | 0.681 | 0.168 | 0.017 | 1.528 | 0.132 | 1.340 | 169.733 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |

## push_lateral_pace_recovery vx=1.50

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.522, trotting=0.347, bounding=0.066, pacing=0.065

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.710 | 0.740 | 0.249 | 0.019 | 1.148 | 0.150 | 1.618 | 187.235 | f=2.62, d=0.50, foot=0.109, width=0.368, pitch=0.000 |
| 2 | trotting | 0.698 | 0.735 | 0.217 | 0.022 | 1.189 | 0.169 | 1.278 | 161.357 | f=3.38, d=0.50, foot=0.080, width=0.292, pitch=0.038 |
| 3 | bounding | 0.648 | 0.752 | 0.262 | 0.018 | 1.283 | 0.140 | 1.538 | 198.087 | f=3.00, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | pacing | 0.648 | 0.698 | 0.264 | 0.022 | 1.122 | 0.123 | 1.354 | 177.114 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.038 |

## push_lateral_pace_recovery vx=1.80

- target_gait_from_task_map: `pacing`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.416, trotting=0.374, bounding=0.133, pacing=0.077

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.662 | 0.725 | 0.366 | 0.024 | 1.275 | 0.179 | 1.699 | 186.419 | f=2.62, d=0.50, foot=0.109, width=0.292, pitch=0.000 |
| 2 | trotting | 0.658 | 0.740 | 0.302 | 0.023 | 1.107 | 0.164 | 1.458 | 171.536 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=0.000 |
| 3 | bounding | 0.627 | 0.690 | 0.331 | 0.026 | 1.220 | 0.152 | 1.547 | 202.126 | f=3.38, d=0.50, foot=0.091, width=0.342, pitch=-0.038 |
| 4 | pacing | 0.611 | 0.694 | 0.322 | 0.021 | 1.301 | 0.124 | 1.421 | 178.240 | f=2.88, d=0.50, foot=0.120, width=0.342, pitch=0.000 |

## ramp_up_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.423, trotting=0.342, pacing=0.204, bounding=0.031

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.892 | 0.799 | 0.085 | 0.007 | 1.219 | 0.053 | 1.187 | 239.557 | f=3.00, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.885 | 0.751 | 0.069 | 0.009 | 1.657 | 0.065 | 0.957 | 166.838 | f=2.61, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 3 | pacing | 0.870 | 0.766 | 0.093 | 0.008 | 1.226 | 0.015 | 0.603 | 222.966 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.813 | 0.713 | 0.146 | 0.008 | 1.847 | 0.059 | 1.119 | 258.957 | f=2.61, d=0.50, foot=0.090, width=0.341, pitch=0.000 |

## ramp_up_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.488, trotting=0.359, pacing=0.120, bounding=0.033

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.854 | 0.751 | 0.117 | 0.018 | 1.518 | 0.051 | 1.352 | 170.669 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 2 | trotting | 0.845 | 0.720 | 0.129 | 0.012 | 1.788 | 0.045 | 1.292 | 164.034 | f=2.61, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.812 | 0.826 | 0.150 | 0.016 | 1.521 | 0.041 | 1.183 | 174.451 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.773 | 0.680 | 0.171 | 0.014 | 1.798 | 0.044 | 1.320 | 221.114 | f=3.39, d=0.50, foot=0.120, width=0.380, pitch=0.039 |

## ramp_up_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.522, trotting=0.335, pacing=0.115, bounding=0.028

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.776 | 0.723 | 0.248 | 0.021 | 1.686 | 0.047 | 1.719 | 189.427 | f=3.00, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 2 | trotting | 0.763 | 0.709 | 0.246 | 0.019 | 1.633 | 0.046 | 1.217 | 164.122 | f=3.39, d=0.50, foot=0.051, width=0.330, pitch=0.039 |
| 3 | pacing | 0.731 | 0.728 | 0.288 | 0.022 | 1.376 | 0.027 | 1.243 | 170.224 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.689 | 0.687 | 0.285 | 0.022 | 1.585 | 0.034 | 1.419 | 188.284 | f=3.00, d=0.50, foot=0.091, width=0.380, pitch=0.000 |

## ramp_up_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.606, trotting=0.315, pacing=0.067, bounding=0.012

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.655 | 0.713 | 0.421 | 0.023 | 1.504 | 0.056 | 1.883 | 198.491 | f=3.00, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | trotting | 0.635 | 0.699 | 0.446 | 0.022 | 1.421 | 0.050 | 1.453 | 179.367 | f=3.00, d=0.50, foot=0.051, width=0.292, pitch=0.000 |
| 3 | pacing | 0.589 | 0.649 | 0.485 | 0.029 | 1.387 | 0.037 | 1.626 | 204.289 | f=2.88, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 4 | bounding | 0.537 | 0.640 | 0.496 | 0.024 | 1.364 | 0.039 | 1.823 | 208.179 | f=3.00, d=0.50, foot=0.091, width=0.342, pitch=-0.038 |

## rough_slope_trot_robustness vx=0.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.513, trotting=0.360, pacing=0.097, bounding=0.030

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.842 | 0.745 | 0.085 | 0.006 | 1.507 | 0.067 | 1.058 | 226.497 | f=3.39, d=0.50, foot=0.050, width=0.330, pitch=0.039 |
| 2 | trotting | 0.831 | 0.730 | 0.085 | 0.005 | 1.734 | 0.077 | 0.966 | 160.533 | f=2.61, d=0.50, foot=0.050, width=0.291, pitch=0.039 |
| 3 | pacing | 0.791 | 0.683 | 0.098 | 0.008 | 1.773 | 0.024 | 0.764 | 200.622 | f=2.50, d=0.50, foot=0.120, width=0.340, pitch=0.040 |
| 4 | bounding | 0.756 | 0.664 | 0.124 | 0.008 | 1.441 | 0.083 | 0.974 | 234.439 | f=3.39, d=0.50, foot=0.090, width=0.380, pitch=0.039 |

## rough_slope_trot_robustness vx=1.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.493, trotting=0.356, pacing=0.113, bounding=0.039

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.783 | 0.773 | 0.171 | 0.015 | 1.675 | 0.056 | 1.403 | 177.300 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
| 2 | trotting | 0.773 | 0.700 | 0.165 | 0.018 | 1.339 | 0.058 | 1.241 | 173.124 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.039 |
| 3 | pacing | 0.739 | 0.650 | 0.175 | 0.016 | 1.563 | 0.068 | 1.304 | 172.418 | f=2.89, d=0.50, foot=0.120, width=0.341, pitch=0.039 |
| 4 | bounding | 0.707 | 0.656 | 0.185 | 0.019 | 1.267 | 0.070 | 1.057 | 177.217 | f=3.39, d=0.50, foot=0.091, width=0.380, pitch=0.039 |

## rough_slope_trot_robustness vx=1.50

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `pronking`
- soft_distribution_from_best_per_gait: pronking=0.384, trotting=0.288, pacing=0.212, bounding=0.116

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pronking | 0.670 | 0.675 | 0.313 | 0.020 | 1.539 | 0.073 | 1.378 | 173.789 | f=3.39, d=0.50, foot=0.109, width=0.291, pitch=0.039 |
| 2 | trotting | 0.661 | 0.677 | 0.323 | 0.023 | 1.034 | 0.068 | 1.258 | 173.285 | f=3.38, d=0.50, foot=0.080, width=0.368, pitch=0.038 |
| 3 | pacing | 0.652 | 0.684 | 0.293 | 0.022 | 1.336 | 0.058 | 1.429 | 179.268 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | bounding | 0.634 | 0.647 | 0.301 | 0.018 | 1.392 | 0.057 | 1.537 | 202.107 | f=3.39, d=0.50, foot=0.120, width=0.341, pitch=0.039 |

## rough_slope_trot_robustness vx=2.00

- target_gait_from_task_map: `trotting`
- best_gait_by_neutral_score: `trotting`
- soft_distribution_from_best_per_gait: trotting=0.715, pacing=0.247, pronking=0.024, bounding=0.013

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | trotting | 0.543 | 0.647 | 0.493 | 0.024 | 1.352 | 0.072 | 1.559 | 189.164 | f=2.62, d=0.50, foot=0.109, width=0.330, pitch=0.038 |
| 2 | pacing | 0.511 | 0.619 | 0.541 | 0.023 | 1.366 | 0.070 | 1.646 | 211.752 | f=2.50, d=0.50, foot=0.120, width=0.342, pitch=0.000 |
| 3 | pronking | 0.441 | 0.644 | 0.649 | 0.024 | 1.293 | 0.073 | 1.634 | 214.207 | f=3.00, d=0.50, foot=0.080, width=0.368, pitch=-0.038 |
| 4 | bounding | 0.424 | 0.603 | 0.641 | 0.017 | 1.657 | 0.064 | 1.655 | 243.284 | f=3.00, d=0.50, foot=0.120, width=0.380, pitch=-0.038 |

## stepping_stones_easy_bound_highspeed vx=1.70

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.319, pronking=0.294, trotting=0.204, bounding=0.183

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.595 | 0.664 | 0.462 | 0.019 | 1.388 | 0.113 | 1.448 | 222.920 | f=2.89, d=0.50, foot=0.120, width=0.380, pitch=0.039 |
| 2 | pronking | 0.593 | 0.684 | 0.470 | 0.019 | 1.608 | 0.117 | 1.600 | 223.382 | f=3.39, d=0.50, foot=0.109, width=0.369, pitch=0.000 |
| 3 | trotting | 0.582 | 0.675 | 0.517 | 0.021 | 1.309 | 0.114 | 1.430 | 222.927 | f=2.61, d=0.50, foot=0.109, width=0.291, pitch=0.000 |
| 4 | bounding | 0.579 | 0.660 | 0.493 | 0.021 | 1.578 | 0.110 | 1.469 | 238.862 | f=2.61, d=0.50, foot=0.120, width=0.341, pitch=0.039 |

## stepping_stones_easy_bound_highspeed vx=2.00

- target_gait_from_task_map: `bounding`
- best_gait_by_neutral_score: `pacing`
- soft_distribution_from_best_per_gait: pacing=0.578, trotting=0.224, bounding=0.121, pronking=0.076

| rank | gait | neutral | live_weighted | vx_err | fall | lateral | scuff | impact | energy | params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | pacing | 0.566 | 0.716 | 0.587 | 0.025 | 1.770 | 0.102 | 1.374 | 211.346 | f=2.50, d=0.50, foot=0.120, width=0.380, pitch=0.038 |
| 2 | trotting | 0.537 | 0.676 | 0.688 | 0.020 | 1.320 | 0.113 | 1.489 | 228.843 | f=2.62, d=0.50, foot=0.109, width=0.330, pitch=0.000 |
| 3 | bounding | 0.519 | 0.713 | 0.686 | 0.018 | 1.238 | 0.103 | 1.597 | 260.045 | f=2.62, d=0.50, foot=0.120, width=0.380, pitch=0.000 |
| 4 | pronking | 0.505 | 0.676 | 0.759 | 0.020 | 1.573 | 0.121 | 1.548 | 261.752 | f=3.00, d=0.50, foot=0.080, width=0.330, pitch=0.039 |
