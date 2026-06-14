# Fixed-Gait Live Reward Audit

This audit uses the current training reward path, not the offline template objective.

## flat_trot_efficiency vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.913 | 0.913 | 0.063 | 0.010 | 0.959 | 0.935 | 0.909 | 0.500 | 1.000 | 0.080 |
| 2 | bounding | 0.898 | 0.898 | 0.116 | 0.010 | 0.920 | 0.911 | 0.739 | 1.000 | 1.000 | 0.120 |
| 3 | pacing | 0.879 | 0.879 | 0.054 | 0.010 | 0.909 | 0.900 | 0.915 | 1.000 | 1.000 | 0.120 |
| 4 | pronking | 0.858 | 0.858 | 0.061 | 0.009 | 0.960 | 0.926 | 0.878 | 0.500 | 1.000 | 0.080 |

## flat_trot_efficiency vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.890 | 0.890 | 0.122 | 0.018 | 0.903 | 0.876 | 0.917 | 0.500 | 1.000 | 0.080 |
| 2 | bounding | 0.876 | 0.876 | 0.168 | 0.019 | 0.869 | 0.854 | 0.764 | 1.000 | 1.000 | 0.120 |
| 3 | pacing | 0.860 | 0.860 | 0.135 | 0.019 | 0.844 | 0.838 | 0.922 | 1.000 | 1.000 | 0.120 |
| 4 | pronking | 0.839 | 0.839 | 0.135 | 0.018 | 0.892 | 0.876 | 0.900 | 0.500 | 1.000 | 0.080 |

## flat_trot_efficiency vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.865 | 0.865 | 0.217 | 0.026 | 0.833 | 0.811 | 0.921 | 0.500 | 1.000 | 0.080 |
| 2 | bounding | 0.846 | 0.846 | 0.264 | 0.026 | 0.791 | 0.783 | 0.764 | 1.000 | 1.000 | 0.120 |
| 3 | pacing | 0.829 | 0.829 | 0.262 | 0.026 | 0.754 | 0.761 | 0.906 | 1.000 | 1.000 | 0.120 |
| 4 | pronking | 0.820 | 0.820 | 0.274 | 0.024 | 0.787 | 0.820 | 0.895 | 0.500 | 1.000 | 0.080 |

## flat_trot_efficiency vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.840 | 0.840 | 0.389 | 0.032 | 0.728 | 0.754 | 0.916 | 0.500 | 1.000 | 0.080 |
| 2 | bounding | 0.804 | 0.804 | 0.459 | 0.032 | 0.629 | 0.732 | 0.722 | 1.000 | 1.000 | 0.120 |
| 3 | pacing | 0.800 | 0.800 | 0.428 | 0.032 | 0.654 | 0.693 | 0.883 | 1.000 | 1.000 | 0.120 |
| 4 | pronking | 0.794 | 0.794 | 0.485 | 0.030 | 0.632 | 0.770 | 0.881 | 0.500 | 1.000 | 0.080 |

## ramp_up_trot_robustness vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.906 | 0.906 | 0.075 | 0.010 | 0.948 | 0.921 | 0.865 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.905 | 0.905 | 0.079 | 0.009 | 0.942 | 0.925 | 0.880 | 0.500 | 1.000 | 0.080 |
| 3 | pacing | 0.880 | 0.880 | 0.102 | 0.009 | 0.879 | 0.899 | 0.888 | 1.000 | 1.000 | 0.120 |
| 4 | bounding | 0.878 | 0.878 | 0.117 | 0.009 | 0.915 | 0.896 | 0.715 | 1.000 | 1.000 | 0.120 |

## ramp_up_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.887 | 0.887 | 0.137 | 0.018 | 0.887 | 0.866 | 0.870 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.881 | 0.881 | 0.156 | 0.018 | 0.863 | 0.863 | 0.860 | 0.500 | 1.000 | 0.080 |
| 3 | pacing | 0.860 | 0.860 | 0.163 | 0.019 | 0.815 | 0.833 | 0.873 | 1.000 | 1.000 | 0.120 |
| 4 | bounding | 0.859 | 0.859 | 0.196 | 0.018 | 0.835 | 0.836 | 0.713 | 1.000 | 1.000 | 0.120 |

## ramp_up_trot_robustness vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.860 | 0.860 | 0.284 | 0.025 | 0.778 | 0.814 | 0.827 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.854 | 0.854 | 0.285 | 0.024 | 0.772 | 0.802 | 0.836 | 0.500 | 1.000 | 0.080 |
| 3 | pacing | 0.833 | 0.833 | 0.295 | 0.025 | 0.724 | 0.765 | 0.833 | 1.000 | 1.000 | 0.120 |
| 4 | bounding | 0.832 | 0.832 | 0.321 | 0.026 | 0.732 | 0.772 | 0.688 | 1.000 | 1.000 | 0.120 |

## ramp_up_trot_robustness vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.823 | 0.823 | 0.496 | 0.030 | 0.652 | 0.745 | 0.806 | 0.500 | 1.000 | 0.080 |
| 2 | pronking | 0.820 | 0.820 | 0.515 | 0.030 | 0.617 | 0.769 | 0.757 | 0.500 | 1.000 | 0.080 |
| 3 | pacing | 0.795 | 0.795 | 0.524 | 0.030 | 0.585 | 0.705 | 0.768 | 1.000 | 1.000 | 0.120 |
| 4 | bounding | 0.794 | 0.794 | 0.552 | 0.031 | 0.566 | 0.715 | 0.632 | 1.000 | 1.000 | 0.120 |

## rough_slope_trot_robustness vx=0.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.860 | 0.860 | 0.102 | 0.010 | 0.922 | 0.922 | 0.791 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.850 | 0.850 | 0.092 | 0.009 | 0.933 | 0.928 | 0.781 | 0.500 | 1.000 | 0.080 |
| 3 | bounding | 0.822 | 0.822 | 0.138 | 0.010 | 0.889 | 0.900 | 0.673 | 1.000 | 1.000 | 0.120 |
| 4 | pacing | 0.820 | 0.820 | 0.111 | 0.008 | 0.873 | 0.899 | 0.832 | 1.000 | 1.000 | 0.120 |

## rough_slope_trot_robustness vx=1.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.833 | 0.833 | 0.200 | 0.018 | 0.824 | 0.858 | 0.763 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.822 | 0.822 | 0.191 | 0.017 | 0.829 | 0.851 | 0.756 | 0.500 | 1.000 | 0.080 |
| 3 | bounding | 0.805 | 0.805 | 0.225 | 0.020 | 0.793 | 0.827 | 0.658 | 1.000 | 1.000 | 0.120 |
| 4 | pacing | 0.802 | 0.802 | 0.184 | 0.019 | 0.797 | 0.830 | 0.803 | 1.000 | 1.000 | 0.120 |

## rough_slope_trot_robustness vx=1.50

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `pronking`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pronking | 0.788 | 0.788 | 0.397 | 0.024 | 0.660 | 0.784 | 0.692 | 0.500 | 1.000 | 0.080 |
| 2 | trotting | 0.786 | 0.786 | 0.358 | 0.023 | 0.680 | 0.761 | 0.730 | 0.500 | 1.000 | 0.080 |
| 3 | bounding | 0.766 | 0.766 | 0.409 | 0.026 | 0.628 | 0.751 | 0.606 | 1.000 | 1.000 | 0.120 |
| 4 | pacing | 0.765 | 0.765 | 0.340 | 0.025 | 0.667 | 0.741 | 0.740 | 1.000 | 1.000 | 0.120 |

## rough_slope_trot_robustness vx=2.00

- target_gait: `trotting`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.736 | 0.736 | 0.650 | 0.028 | 0.491 | 0.686 | 0.645 | 0.500 | 1.000 | 0.080 |
| 2 | pronking | 0.728 | 0.728 | 0.806 | 0.028 | 0.414 | 0.724 | 0.587 | 0.500 | 1.000 | 0.080 |
| 3 | pacing | 0.709 | 0.709 | 0.693 | 0.029 | 0.452 | 0.668 | 0.622 | 1.000 | 1.000 | 0.120 |
| 4 | bounding | 0.706 | 0.706 | 0.848 | 0.030 | 0.373 | 0.699 | 0.491 | 1.000 | 1.000 | 0.120 |

## push_lateral_pace_recovery vx=1.50

- target_gait: `pacing`
- live_best_by_weighted_metric_reward: `trotting`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trotting | 0.781 | 0.781 | 0.443 | 0.022 | 0.650 | 0.776 | 0.804 | 0.500 | 1.000 | 0.080 |
| 2 | pronking | 0.773 | 0.773 | 0.537 | 0.019 | 0.572 | 0.778 | 0.735 | 0.500 | 1.000 | 0.080 |
| 3 | bounding | 0.766 | 0.766 | 0.443 | 0.021 | 0.645 | 0.763 | 0.672 | 1.000 | 1.000 | 0.120 |
| 4 | pacing | 0.758 | 0.758 | 0.457 | 0.022 | 0.603 | 0.716 | 0.801 | 1.000 | 1.000 | 0.120 |

## stepping_stones_easy_bound_highspeed vx=2.00

- target_gait: `bounding`
- live_best_by_weighted_metric_reward: `pacing`

| rank | gait | weighted | reward | vx_err | done | progress | slip | orientation | clearance | actual | foot |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pacing | 0.806 | 0.806 | 0.633 | 0.027 | 0.475 | 0.623 | 0.734 | 1.000 | 1.000 | 0.120 |
| 2 | bounding | 0.797 | 0.797 | 0.741 | 0.025 | 0.402 | 0.648 | 0.592 | 1.000 | 1.000 | 0.120 |
| 3 | trotting | 0.758 | 0.758 | 0.819 | 0.023 | 0.385 | 0.642 | 0.678 | 0.500 | 1.000 | 0.080 |
| 4 | pronking | 0.755 | 0.755 | 0.849 | 0.024 | 0.338 | 0.657 | 0.655 | 0.500 | 1.000 | 0.080 |
