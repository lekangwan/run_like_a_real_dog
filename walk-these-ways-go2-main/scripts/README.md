# `scripts/` 高层代码索引

本目录保留项目真实实验链。文件较多，是因为奖励审查、步态公平性检查、信息通路诊断、
历史训练方案和汇报工具都曾在这里迭代。**日常开发不需要从头阅读全部文件。**

项目已于 2026-08-31 进入归档状态。这里的“主线”表示最终结果实际使用的代码链，
不表示仍在继续训练。归档和复现边界见 `../ARCHIVE_AND_REPRODUCIBILITY.md`。

## 一、最终高层实验链

建议按以下顺序阅读：

| 文件 | 作用 | 是否常用 |
|---|---|---|
| `gait_project_config.py` | 集中保存主线任务表、模板和公共路径 | 是 |
| `gait_conditions.py` | 根据任务条件配置平地、坡面、粗糙地形、推扰和踏石 | 是 |
| `train_high_level_ppo.py` | 高层 actor、critic、底层加载和 PPO 公共实现 | 是 |
| `train_high_level_oracle_ppo.py` | 当前真实高层训练入口，组装师生表示、步态选择和训练循环 | 是 |
| `evaluate_high_level_policy_by_task.py` | 按地形和速度独立评测高层检查点 | 是 |
| `analyze_adaptive_vs_forced_trot.py` | 汇总自适应策略与固定小跑的多随机种子成对结果 | 是 |
| `play_oracle_policy_training_map.py` | 单环境可视化策略行为 | 按需 |
| `record_high_level_policy_videos.py` | 录制高层策略视频 | 按需 |

最终汇报结果来自这套原始实验链，而不是 `high_level_minimal/`。

### 原则上必须保留的最小集合

```text
gait_project_config.py
gait_conditions.py
train_high_level_ppo.py
train_high_level_oracle_ppo.py
evaluate_high_level_policy_by_task.py
analyze_adaptive_vs_forced_trot.py
```

依赖方向是：配置和地形条件先被训练入口加载；训练入口复用早期高层公共实现；评测
入口再导入训练入口中的环境和模型定义；分析脚本最后读取评测 CSV。这里不是六个彼此
独立的程序，删除中间任意一层都可能让后续脚本无法导入。

## 二、主线验证工具

这些文件不参与正式训练，但用于判断奖励和比较协议是否可信：

| 文件 | 回答的问题 |
|---|---|
| `evaluate_paired_gait_live_reward.py` | 相同初始状态下，两种强制步态谁更好 |
| `check_high_level_reward_consistency.py` | 在线奖励与离线重算是否使用相同公式 |
| `evaluate_gait_target_fairness.py` | 每种步态获得相同连续参数搜索预算后如何排名 |
| `evaluate_fixed_gait_live_reward.py` | 固定步态在真实仿真轨迹上的奖励表现 |
| `evaluate_gait_templates.py` | 检查 WTW 步态模板的基本可执行性 |
| `analyze_wtw_capability_frontier.py` | 判断冻结 WTW 能否在指标上支持高层改进 |

只有在修改奖励公式、步态模板或评测协议时，才需要重新阅读这些脚本。

## 三、历史奖励分析工具

以下文件记录了从早期线性奖励到统一物理奖励的审查过程。它们用于复现实验，不是最终
训练入口：

```text
analyze_fixed_gait_reward_gaps.py
analyze_metric_sanity_audit.py
offline_rescore_unified_reward.py
rescore_fair_grid_live_profiles.py
select_heldout_gait_configs.py
select_metric_sanity_configs.py
```

## 四、历史信息通路与步态先验实验

这些脚本用于回答“历史信息是否存在、师生隐变量是否编码、选择器是否使用”等问题，
以及复现已经放弃的显式步态参考方案：

```text
collect_high_level_info_path_data.py
analyze_high_level_info_path.py
build_soft_selector_targets.py
visualize_oracle_training_results.py
```

`build_soft_selector_targets.py` 生成的是历史诊断用步态参考，不属于当前“无任务编号、
无人工步态答案”的最终方案。

## 五、历史训练与专项实验

| 文件 | 说明 |
|---|---|
| `train_high_level_ppo.py` | 早期通用高层实现，同时仍被当前训练入口复用 |
| `test_gait_conditioned_residuals.py` | 检查旧版步态条件连续参数分支 |
| `test_gait_input_residuals.py` | 检查新版“状态加已选步态”连续参数分支 |
| `test_oracle_policy_route.py` | 连续路线和地形切换专项测试 |
| `play_task_gait_oracle.py` | 强制任务步态的历史可视化工具 |
| `play_training_scenes_oracle.py` | 多训练场景历史可视化工具 |
| `finetune_low_level_gait_transitions.py` | 底层步态切换专项微调，不属于高层主训练 |

## 六、原 WTW 与执行器工具

```text
train.py
play.py
actuator_net/
```

这些属于原 WTW 训练和执行器网络基础设施。高层整理时不要随意移动或修改。

## 七、汇报生成工具

```text
make_report_core_figure.py
```

PPT 专用脚本和结果统一放在 `ppt_assets/`，阶段汇报文件放在 `reports/`。

## 八、为什么暂时不直接移动文件

历史脚本普遍使用类似下面的同目录导入：

```python
from train_high_level_oracle_ppo import ...
from gait_project_config import ...
```

大量运行命令和实验记录也直接引用 `scripts/<name>.py`。现在直接移动会破坏导入路径和
历史复现。若未来确实需要物理迁移，应先把公共实现改成正式 Python 包，再统一更新
命令；在此之前，本索引承担逻辑分类作用。项目归档后不值得只为减少文件数量而进行这项
高风险重构。
