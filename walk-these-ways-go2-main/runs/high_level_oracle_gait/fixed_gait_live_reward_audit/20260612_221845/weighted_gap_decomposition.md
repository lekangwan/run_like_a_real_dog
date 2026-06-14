# Fixed-Gait Reward Gap Decomposition

Positive weighted_gap means the metric helps the target gait. Negative weighted_gap means it helps the competitor.

## flat_trot_efficiency vx=0.50: trotting vs bounding

- target_reward: 0.9132
- competitor_reward: 0.8979
- target_minus_competitor: 0.0152

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| orientation | 0.3000 | 0.9091 | 0.7390 | 0.1701 | 0.0057 |
| progress | 1.0000 | 0.9594 | 0.9200 | 0.0394 | 0.0044 |
| slip | 1.2000 | 0.9353 | 0.9105 | 0.0248 | 0.0033 |
| vertical_bounce | 0.8000 | 0.8764 | 0.8455 | 0.0309 | 0.0028 |
| yaw_tracking | 0.3000 | 0.8111 | 0.8674 | -0.0564 | -0.0019 |
| action_smoothness | 0.7000 | 1.0000 | 0.9900 | 0.0100 | 0.0008 |
| gait_stability | 0.4000 | 1.0000 | 0.9912 | 0.0088 | 0.0004 |
| lateral_drift | 0.8000 | 0.4102 | 0.4142 | -0.0041 | -0.0004 |
| survival | 2.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## flat_trot_efficiency vx=1.00: trotting vs bounding

- target_reward: 0.8904
- competitor_reward: 0.8756
- target_minus_competitor: 0.0148

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| orientation | 0.3000 | 0.9166 | 0.7642 | 0.1524 | 0.0051 |
| progress | 1.0000 | 0.9030 | 0.8688 | 0.0342 | 0.0038 |
| slip | 1.2000 | 0.8763 | 0.8535 | 0.0228 | 0.0031 |
| vertical_bounce | 0.8000 | 0.8045 | 0.7820 | 0.0225 | 0.0020 |
| action_smoothness | 0.7000 | 1.0000 | 0.9816 | 0.0184 | 0.0014 |
| yaw_tracking | 0.3000 | 0.7542 | 0.7884 | -0.0342 | -0.0012 |
| gait_stability | 0.4000 | 1.0000 | 0.9839 | 0.0161 | 0.0007 |
| lateral_drift | 0.8000 | 0.4069 | 0.4099 | -0.0029 | -0.0003 |
| survival | 2.0000 | 1.0000 | 0.9999 | 0.0001 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## flat_trot_efficiency vx=1.50: trotting vs bounding

- target_reward: 0.8649
- competitor_reward: 0.8462
- target_minus_competitor: 0.0187

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| vertical_bounce | 0.8000 | 0.7542 | 0.6944 | 0.0598 | 0.0054 |
| orientation | 0.3000 | 0.9215 | 0.7637 | 0.1578 | 0.0053 |
| progress | 1.0000 | 0.8335 | 0.7913 | 0.0422 | 0.0047 |
| slip | 1.2000 | 0.8107 | 0.7829 | 0.0278 | 0.0037 |
| lateral_drift | 0.8000 | 0.3862 | 0.4113 | -0.0252 | -0.0023 |
| action_smoothness | 0.7000 | 1.0000 | 0.9744 | 0.0256 | 0.0020 |
| yaw_tracking | 0.3000 | 0.6753 | 0.7131 | -0.0378 | -0.0013 |
| gait_stability | 0.4000 | 1.0000 | 0.9776 | 0.0224 | 0.0010 |
| survival | 2.0000 | 1.0000 | 0.9999 | 0.0001 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## flat_trot_efficiency vx=2.00: trotting vs bounding

- target_reward: 0.8400
- competitor_reward: 0.8036
- target_minus_competitor: 0.0364

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| vertical_bounce | 0.8000 | 0.7343 | 0.5539 | 0.1803 | 0.0162 |
| progress | 1.0000 | 0.7279 | 0.6286 | 0.0993 | 0.0112 |
| orientation | 0.3000 | 0.9158 | 0.7222 | 0.1936 | 0.0065 |
| yaw_tracking | 0.3000 | 0.5624 | 0.6543 | -0.0919 | -0.0031 |
| slip | 1.2000 | 0.7544 | 0.7316 | 0.0229 | 0.0031 |
| action_smoothness | 0.7000 | 1.0000 | 0.9689 | 0.0311 | 0.0024 |
| gait_stability | 0.4000 | 1.0000 | 0.9728 | 0.0272 | 0.0012 |
| lateral_drift | 0.8000 | 0.3894 | 0.4026 | -0.0132 | -0.0012 |
| survival | 2.0000 | 1.0000 | 0.9999 | 0.0001 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## push_lateral_pace_recovery vx=1.50: pacing vs trotting

- target_reward: 0.7583
- competitor_reward: 0.7814
- target_minus_competitor: -0.0231

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| lateral_drift | 1.6000 | 0.2068 | 0.3138 | -0.1070 | -0.0186 |
| progress | 1.0000 | 0.6029 | 0.6500 | -0.0471 | -0.0051 |
| yaw_rate | 0.6000 | 0.7140 | 0.6816 | 0.0324 | 0.0021 |
| action_smoothness | 0.7000 | 0.9781 | 1.0000 | -0.0219 | -0.0017 |
| yaw_tracking | 0.3000 | 0.5875 | 0.5486 | 0.0389 | 0.0013 |
| gait_stability | 0.4000 | 0.9808 | 1.0000 | -0.0192 | -0.0008 |
| orientation | 1.2000 | 0.8013 | 0.8036 | -0.0023 | -0.0003 |
| survival | 2.0000 | 0.9997 | 0.9995 | 0.0002 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## ramp_up_trot_robustness vx=0.50: trotting vs pronking

- target_reward: 0.9053
- competitor_reward: 0.9058
- target_minus_competitor: -0.0005

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| yaw_tracking | 0.3000 | 0.7879 | 0.8720 | -0.0842 | -0.0029 |
| orientation | 0.8000 | 0.8801 | 0.8647 | 0.0154 | 0.0014 |
| action_smoothness | 0.7000 | 1.0000 | 0.9903 | 0.0097 | 0.0008 |
| progress | 1.0000 | 0.9420 | 0.9483 | -0.0063 | -0.0007 |
| slip | 1.2000 | 0.9252 | 0.9205 | 0.0047 | 0.0006 |
| gait_stability | 0.4000 | 1.0000 | 0.9915 | 0.0085 | 0.0004 |
| lateral_drift | 0.8000 | 0.3661 | 0.3675 | -0.0014 | -0.0001 |
| survival | 2.0000 | 1.0000 | 1.0000 | -0.0000 | -0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## ramp_up_trot_robustness vx=1.00: trotting vs pronking

- target_reward: 0.8814
- competitor_reward: 0.8874
- target_minus_competitor: -0.0060

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| progress | 1.0000 | 0.8627 | 0.8869 | -0.0242 | -0.0028 |
| yaw_tracking | 0.3000 | 0.7024 | 0.7814 | -0.0790 | -0.0028 |
| action_smoothness | 0.7000 | 1.0000 | 0.9821 | 0.0179 | 0.0015 |
| lateral_drift | 0.8000 | 0.3545 | 0.3673 | -0.0128 | -0.0012 |
| orientation | 0.8000 | 0.8601 | 0.8702 | -0.0101 | -0.0009 |
| gait_stability | 0.4000 | 1.0000 | 0.9844 | 0.0156 | 0.0007 |
| slip | 1.2000 | 0.8625 | 0.8659 | -0.0033 | -0.0005 |
| survival | 2.0000 | 0.9999 | 1.0000 | -0.0001 | -0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## ramp_up_trot_robustness vx=1.50: trotting vs pronking

- target_reward: 0.8536
- competitor_reward: 0.8596
- target_minus_competitor: -0.0061

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| lateral_drift | 0.8000 | 0.3234 | 0.3700 | -0.0466 | -0.0043 |
| yaw_tracking | 0.3000 | 0.5978 | 0.6881 | -0.0903 | -0.0031 |
| action_smoothness | 0.7000 | 1.0000 | 0.9754 | 0.0246 | 0.0020 |
| slip | 1.2000 | 0.8016 | 0.8141 | -0.0126 | -0.0018 |
| gait_stability | 0.4000 | 1.0000 | 0.9785 | 0.0215 | 0.0010 |
| orientation | 0.8000 | 0.8355 | 0.8266 | 0.0089 | 0.0008 |
| progress | 1.0000 | 0.7724 | 0.7782 | -0.0058 | -0.0007 |
| survival | 2.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## ramp_up_trot_robustness vx=2.00: trotting vs pronking

- target_reward: 0.8227
- competitor_reward: 0.8200
- target_minus_competitor: 0.0027

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| orientation | 0.8000 | 0.8060 | 0.7569 | 0.0491 | 0.0046 |
| progress | 1.0000 | 0.6521 | 0.6169 | 0.0353 | 0.0041 |
| slip | 1.2000 | 0.7452 | 0.7695 | -0.0242 | -0.0034 |
| lateral_drift | 0.8000 | 0.2954 | 0.3291 | -0.0337 | -0.0031 |
| yaw_tracking | 0.3000 | 0.4938 | 0.5823 | -0.0885 | -0.0031 |
| action_smoothness | 0.7000 | 1.0000 | 0.9701 | 0.0299 | 0.0024 |
| gait_stability | 0.4000 | 1.0000 | 0.9738 | 0.0262 | 0.0012 |
| survival | 2.0000 | 0.9998 | 0.9998 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## rough_slope_trot_robustness vx=0.50: trotting vs pronking

- target_reward: 0.8499
- competitor_reward: 0.8601
- target_minus_competitor: -0.0102

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| roll_rate | 0.8000 | 0.4433 | 0.5769 | -0.1336 | -0.0109 |
| lateral_drift | 0.8000 | 0.3495 | 0.3992 | -0.0497 | -0.0041 |
| yaw_tracking | 0.3000 | 0.7038 | 0.6058 | 0.0980 | 0.0030 |
| orientation | 1.2000 | 0.7807 | 0.7906 | -0.0098 | -0.0012 |
| progress | 1.0000 | 0.9331 | 0.9218 | 0.0113 | 0.0012 |
| slip | 1.2000 | 0.9280 | 0.9217 | 0.0063 | 0.0008 |
| action_smoothness | 0.7000 | 1.0000 | 0.9904 | 0.0096 | 0.0007 |
| gait_stability | 0.4000 | 1.0000 | 0.9916 | 0.0084 | 0.0003 |
| survival | 2.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## rough_slope_trot_robustness vx=1.00: trotting vs pronking

- target_reward: 0.8223
- competitor_reward: 0.8331
- target_minus_competitor: -0.0108

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| roll_rate | 0.8000 | 0.4286 | 0.5202 | -0.0916 | -0.0075 |
| lateral_drift | 0.8000 | 0.3359 | 0.3894 | -0.0535 | -0.0044 |
| action_smoothness | 0.7000 | 1.0000 | 0.9823 | 0.0177 | 0.0013 |
| slip | 1.2000 | 0.8510 | 0.8585 | -0.0075 | -0.0009 |
| orientation | 1.2000 | 0.7564 | 0.7630 | -0.0067 | -0.0008 |
| gait_stability | 0.4000 | 1.0000 | 0.9845 | 0.0155 | 0.0006 |
| progress | 1.0000 | 0.8286 | 0.8243 | 0.0043 | 0.0004 |
| yaw_tracking | 0.3000 | 0.6324 | 0.6191 | 0.0134 | 0.0004 |
| survival | 2.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## rough_slope_trot_robustness vx=1.50: trotting vs pronking

- target_reward: 0.7858
- competitor_reward: 0.7880
- target_minus_competitor: -0.0022

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| orientation | 1.2000 | 0.7298 | 0.6919 | 0.0379 | 0.0046 |
| lateral_drift | 0.8000 | 0.3080 | 0.3645 | -0.0565 | -0.0046 |
| slip | 1.2000 | 0.7606 | 0.7837 | -0.0231 | -0.0028 |
| roll_rate | 0.8000 | 0.4140 | 0.4457 | -0.0317 | -0.0026 |
| progress | 1.0000 | 0.6795 | 0.6597 | 0.0198 | 0.0020 |
| action_smoothness | 0.7000 | 1.0000 | 0.9763 | 0.0237 | 0.0017 |
| yaw_tracking | 0.3000 | 0.5173 | 0.5634 | -0.0461 | -0.0014 |
| gait_stability | 0.4000 | 1.0000 | 0.9793 | 0.0207 | 0.0008 |
| survival | 2.0000 | 1.0000 | 0.9998 | 0.0002 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## rough_slope_trot_robustness vx=2.00: trotting vs pronking

- target_reward: 0.7360
- competitor_reward: 0.7280
- target_minus_competitor: 0.0079

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| progress | 1.0000 | 0.4908 | 0.4143 | 0.0765 | 0.0078 |
| orientation | 1.2000 | 0.6453 | 0.5869 | 0.0584 | 0.0072 |
| slip | 1.2000 | 0.6857 | 0.7241 | -0.0384 | -0.0047 |
| lateral_drift | 0.8000 | 0.2905 | 0.3370 | -0.0465 | -0.0038 |
| action_smoothness | 0.7000 | 1.0000 | 0.9723 | 0.0277 | 0.0020 |
| gait_stability | 0.4000 | 1.0000 | 0.9758 | 0.0242 | 0.0010 |
| yaw_tracking | 0.3000 | 0.4052 | 0.4367 | -0.0315 | -0.0010 |
| roll_rate | 0.8000 | 0.3391 | 0.3474 | -0.0084 | -0.0007 |
| survival | 2.0000 | 0.9996 | 0.9989 | 0.0007 | 0.0002 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## stepping_stones_easy_bound_highspeed vx=2.00: bounding vs pacing

- target_reward: 0.7966
- competitor_reward: 0.8059
- target_minus_competitor: -0.0094

| metric | weight | target_score | competitor_score | raw_gap | weighted_gap |
|---|---:|---:|---:|---:|---:|
| orientation | 0.6000 | 0.5921 | 0.7336 | -0.1415 | -0.0109 |
| lateral_drift | 0.8000 | 0.3176 | 0.2257 | 0.0918 | 0.0094 |
| progress | 1.0000 | 0.4016 | 0.4746 | -0.0730 | -0.0094 |
| yaw_tracking | 0.3000 | 0.4316 | 0.3991 | 0.0325 | 0.0013 |
| action_smoothness | 0.7000 | 0.9751 | 0.9731 | 0.0020 | 0.0002 |
| gait_stability | 0.4000 | 0.9782 | 0.9764 | 0.0018 | 0.0001 |
| survival | 2.0000 | 0.9996 | 0.9998 | -0.0002 | -0.0001 |
| clearance | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_magnitude | 0.6000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| action_boundary_margin | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
