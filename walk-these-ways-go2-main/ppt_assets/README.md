# 保研面试 PPT 素材说明

本目录只包含汇报素材与 CPU 生成脚本，没有修改训练、环境或控制代码。

## 素材索引

| 文件 | 用途 | 数据来源与限制 |
|---|---|---|
| `p2_system_framework.png` | 方法框架图 | 按 2026-07-21 汇报主模型绘制。连续参数调节以虚线标为实验分支；主模型实际固定默认连续参数。 |
| `p2_isaac_gym_scene.png` | Isaac Gym 场景截图 | 从已有上坡 1.5 m/s、1080p 单环境录像第 24 帧提取。仅展示仿真环境。 |
| `p2_terrain_gallery.png` | 四种仿真地形总览 | 使用仓库已有的真实高度场预览，包含平地、连续上坡、粗糙坡面和踏石；图中明确区分训练场景与未见地形测试。 |
| `continuous_route/p2_continuous_four_terrain.mp4` | 连续四地形穿越演示 | 一条连续赛道依次连接平地、上坡、粗糙坡面和踏石，相机跟随机器人。步态按赛道路段显式切换，用于展示底层多步态执行与过渡能力，不作为自主地形识别证据。 |
| `p2_continuous_four_terrain_0_11s.gif` | 连续路线 GIF | 从上述连续路线视频截取 0 至 11 秒；属于显式步态演示，不作为策略自主选择证据。 |
| `p2_gait_distribution.png` | 四类地形的步态分布 | 平地、上坡为训练场景；粗糙地面、踏石为未见地形。每个点均为 32 个并行环境、1000 步、3 个随机种子的真实选择比例。 |
| `p2_eval_comparison.png` | 自适应策略与固定小跑对比 | 同检查点、同环境、同随机种子的严格成对评测；展示上坡 1.0/1.5 m/s 的速度误差、接触滑移、冲击和机械功率变化。 |
| `p2_ood_generalization.png` | 粗糙地面与踏石测试 | 汇总两个未见地形相对固定小跑的奖励、跟踪、功率和接触指标，并明确展示收益与边界。 |
| `gait_distribution_source.csv` | 步态分布作图数据快照 | 由三个独立评测目录中的 `independent_eval_summary.csv` 聚合。 |
| `eval_comparison_source.csv` | 对比图数据快照 | 原始 `paired_three_seed_summary.csv` 的副本。 |
| `rough_ood_source.csv` | 粗糙地面数据快照 | 三个新随机种子的成对汇总。 |
| `stones_ood_source.csv` | 踏石数据快照 | 三个新随机种子的成对汇总。 |
| `source_manifest.json` | 机器可读来源清单 | 记录检查点、随机种子、数据表和视频来源。 |

## 主模型与数据来源

主检查点：

```text
runs/high_level_oracle_gait/
20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct/
checkpoints/high_level_000049.pt
```

三随机种子成对汇总：

```text
runs/high_level_oracle_gait/
20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct/
independent_eval/paired_three_seed_summary.csv
```

视频来源：

```text
reports/20260721_closure/videos_all_terrains_1080p/
```

场景截图来自上述 1080p 固定相机目录。早期低清跟随视频及其拼接演示已在归档清理时
删除；当前保留连续四地形源视频和对应 GIF。

## 生成命令

在仓库根目录运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib-ppt-assets \
  /home/lekangwan/miniconda3/envs/go2_wtw/bin/python3 \
  ppt_assets/generate_ppt_assets.py
```

脚本只读取已有 CSV 和 1080p MP4，使用 CPU 绘图，不启动 Isaac Gym，也不占用 CUDA。

四地形 GIF 需要重新启动 Isaac Gym 录制，因此由用户手动运行：

```bash
bash ppt_assets/record_four_scene_gif.sh
```

该脚本会顺序录制四段视频，再自动生成 `p2_four_terrain_gaits.gif`。其中步态被明确固定，
不能在 PPT 中称为策略自主切换；自主选择证据应引用 `p2_gait_distribution.png`。

连续四地形路线由用户手动录制：

```bash
bash ppt_assets/record_continuous_four_terrain.sh
```

输出为 `ppt_assets/continuous_route/p2_continuous_four_terrain.mp4`。该视频根据机器人在连续赛道上的
位置依次指定小跑、双脚跳、小跑和跳跃跑，并使用轻微的直线航向校正防止机器人离开长条赛道，
因此只能表述为“连续组合地形上的多步态执行演示”，不能作为自主地形识别或自主导航证据。

## PPT 中建议采用的严谨表述

> 在不使用地形编号、视觉和雷达的条件下，高层策略仅依赖目标速度与本体感知历史，形成了条件和速度相关的步态选择。相对固定小跑，该策略在部分中速上坡场景改善了速度跟踪、接触滑移和落地冲击，但增加了机械功率；高速上坡、未见地形泛化、连续参数稳定增益和实机部署仍未解决。

不要表述为“全地形稳定优于固定步态”，也不要将单环境录像描述为步态切换的证据。

## 仍属于初步实验的部分

- 目前核心结论只覆盖平地与上坡，且来自仿真。
- 连续参数调节没有获得超过仿真重复波动的稳定收益，汇报主模型将其固定为默认值。
- 横向推扰上的未见场景泛化不如固定小跑；踏石只观察到混合利弊，不能声称广泛泛化。
- 新增严格复核表明：粗糙地面多数速度的综合性能不如固定小跑；踏石在跟踪、功率和冲击上有小幅改善，但绝对速度误差仍很大，而且没有形成跳跃步态。
- 统一物理奖励通过了代码一致性检查，但不能证明它是普遍正确的运动质量定义。
- 尚无实机部署视频，目录中的 Go2 画面均来自 Isaac Gym。
