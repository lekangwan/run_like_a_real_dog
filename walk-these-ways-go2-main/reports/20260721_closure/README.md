# 汇报材料索引

## 可直接使用

- `core_result_figure.png`：幻灯片用核心结果图。
- `core_result_figure.pdf`：矢量版本。
- `PRESENTATION_OUTLINE.md`：八页汇报结构和口头说明。

## 视频

`videos_all_terrains_1080p/` 包含五类地形的 1080p、5 秒视频：

| 文件 | 场景 | 速度 | 录制步态 |
|---|---|---:|---|
| `flat_trot_efficiency_vx1p00.mp4` | 平地 | 1.0 m/s | 小跑 |
| `ramp_up_trot_robustness_vx1p50.mp4` | 上坡 | 1.5 m/s | 小跑 |
| `rough_slope_trot_robustness_vx1p00.mp4` | 粗糙坡面 | 1.0 m/s | 小跑 |
| `push_lateral_pace_recovery_vx1p50.mp4` | 横向推扰 | 1.5 m/s | 小跑 |
| `stepping_stones_easy_bound_highspeed_vx2p00.mp4` | 踏石 | 2.0 m/s | 小跑 |

这些视频用于展示仿真场景，不作为步态分化证据。它们采用单环境确定性输出，
而核心定量结果来自 32 个并行环境和三个评测随机种子。尤其是上坡视频中的
小跑不能推翻批量评测中上坡 1.5 m/s 约 75.3% 双脚跳的统计结果，但也不能
拿该视频声称已经直观展示了双脚跳。

## 不建议使用

- `videos/`：早期跟随相机、较低分辨率版本。
- `videos_all_terrains/`：固定相机但分辨率较低的版本。

旧视频保留用于追溯，没有删除。
