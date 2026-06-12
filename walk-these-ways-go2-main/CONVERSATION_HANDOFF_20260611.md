# Conversation Handoff — Go2 High-Level Gait Adaptation (RMA + Reward v4)

Date: 2026-06-11

Read `ACTIVE_PROJECT_CONTEXT.md` and `CONVERSATION_HANDOFF_20260608.md` first.
This document supersedes the "next steps" section of the 0608 handoff.

---

## 1. 整体方案设计

### 1.1 最终目标

冻结的 WTW Go2 低层策略之上，训练一个高层步态适应模块，能：

- 仅从本体感知历史推断地形/环境条件
- 自主选择步态（pronking / trotting / bounding / pacing）
- 连续调整 5 个步态参数（频率、持续时间、足部摆动高度、支撑宽度、身体俯仰角）
- 最终部署到真实 Unitree Go2

### 1.2 两条主线

| 主线 | 内容 | 状态 |
|------|------|:----:|
| **RMA 环境表征** | 用 Teacher-Student 蒸馏让策略从本体感知中隐式学习地形特征 | 已实现，待消融验证 |
| **Reward v4** | 基于 v8 评估数据重新设计 reward 权重，让不同地形真正看重不同的运动质量 | 已改完权重和 CSV，待跑实验 |

### 1.3 为什么不直接用 style reward 或 task one-hot

- `style_reward_scale > 0` 会变成人为指定步态，遇到新地形无法泛化
- `task_onehot` 只在 oracle 验证阶段使用，最终必须移除
- 目标：PPO 自己从 reward 中学会"这个环境条件 → 这个步态更合适"

---

## 2. RMA 架构（已实现）

### 2.1 网络结构

```
训练时：
privileged_obs (14D) → TerrainEncoder (14→128→64→16) → z_teacher (16D)
obs_history (510D)    → AdaptationModule (510→256→128→16) → z_student (16D)

z_input = α · z_teacher + (1-α) · z_student

[obs_history ‖ task_onehot(可选) ‖ z_input] → Backbone (256,256) → Gait(4D) + Residual(5D) + Critic(1D)

loss = PPO + 0.1 · MSE(z_student, z_teacher.detach())

推理时（部署）：
obs_history (510D) → AdaptationModule → z_student (16D)
[obs_history ‖ z_student] → actor → 9D action
```

### 2.2 三阶段 α 退火

```
Phase 1: progress < 25%  → α = 1.0  (纯 z_teacher，稳定训练)
Phase 2: 25% ≤ p < 75%   → α: 1.0 → 0.0 线性退火
Phase 3: progress ≥ 75%  → α = 0.0  (纯 z_student)
```

### 2.3 high_level_privileged_obs (14D)

```
0   terrain_height_mean    measured_heights 均值，clamp
1   terrain_height_std     measured_heights 标准差
2   terrain_height_range   max-min
3   terrain_slope          前后高度梯度代理
4   friction               地面摩擦系数
5   base_mass              附加质量
6-8 com_dx/dy/dz           质心偏移
9   push_active            是否启用扰动
10  push_axis              -1无 / 0纵向 / 1横向
11  body_height            基座离地高度
12  pitch_proxy            投影重力 x
13  roll_proxy             投影重力 y
```

注意：扰动标志来自 `cfg.domain_rand.push_axis_by_env`，是每 env 恒定的配置参数，不是物理传感器。这是 RMA 标准做法。

### 2.4 已添加的命令行参数

```
--z-dim 16            环境隐变量维度（0 = 完全关闭 RMA）
--priv-dim 14         特权观测维度
--adaptation-coef 0.1 蒸馏损失权重
```

### 2.5 改动的文件

| 文件 | 改动 |
|------|------|
| `go2_gym/envs/wrappers/high_level_gait_wrapper.py` | 新增 `get_high_level_privileged_obs()` 方法 |
| `scripts/train_high_level_ppo.py` | `ActorCritic` 新增 `TerrainEncoder`、`AdaptationModule`、`act_student()`；支持 `z_dim=0` bypass |
| `scripts/train_high_level_oracle_ppo.py` | 训练循环接入 RMA 蒸馏；`OracleConditionHighLevelEnv` 新增 `get_base_obs()`、`get_high_level_privileged_obs()` |

---

## 3. Reward v4 重设计（已改完，待实验验证）

### 3.1 为什么重设计

v3 reward 中 progress 在所有 5 个地形都是权重 2.0（cap），导致 PPO 坍缩到 pace（前进效率最高的步态）。v8 评估数据分析发现：
- `trot_contact_style`、`pace_contact_style` 等 token 在 CSV 中定义但从未在代码中实现（被 `.get(token, {})` 静默丢弃）
- 不同地形间的权重差异只在次要指标上（energy ±0.6, clearance ±0.35），无法对抗 progress 2.0 的主导

### 3.2 v4 设计原则

1. progress 不进 focus，只保留 base 1.0（作为底线而非优化目标）
2. 用 v8 数据验证过的"真正能区分步态"的指标作为 focus 主力（最主要是 slip）
3. 不同地形的优化方向应尽可能不同（flat 反 pace、push 顺 pace、stones 推 bound）
4. orientation_stability 拆三级 token 以适应不同地形需求
5. 删除所有无效 contact_style token
6. survival 不进 focus（v8 数据显示 done_rate 在所有步态间无差异）

### 3.3 v4 有效权重表

| 指标 | base | flat | ramp_up | rough_slope | push_lateral | stepping_stones |
|------|:---:|:---:|:------:|:----------:|:----------:|:--------------:|
| progress | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| yaw_tracking | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| orientation | 0.3 | 0.3 | **0.8** | **1.2** | **1.2** | **0.6** |
| lateral_drift | 0.8 | 0.8 | 0.8 | 0.8 | **1.6** | 0.8 |
| **slip** | 0 | **1.2** | **1.2** | **1.2** | 0 | 0 |
| vertical_bounce | 0 | **0.8** | 0 | 0 | 0 | 0 |
| roll_rate | 0 | 0 | 0 | **0.8** | 0 | 0 |
| yaw_rate | 0 | 0 | 0 | 0 | **0.6** | 0 |
| clearance | 0 | 0 | 0 | 0 | 0 | **0.6** |
| gait_stability | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| action_smoothness | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |
| action_magnitude | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| action_boundary_margin | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| survival | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

### 3.4 CSV reward_focus（v4）

```
flat_trot_efficiency:                low_slip,low_vertical_bounce
ramp_up_trot_robustness:             low_slip,orientation_stability
rough_slope_trot_robustness:         low_slip,orientation_stability_strong,low_roll_rate
push_lateral_pace_recovery:          orientation_stability_strong,low_yaw_rate,low_lateral_drift
stepping_stones_easy_bound_highspeed: foot_clearance,orientation_stability_mild
```

### 3.5 TASK_REWARD_FOCUS_WEIGHTS（v4 代码）

```python
TASK_REWARD_FOCUS_WEIGHTS = {
    "low_slip": {"slip": 1.2},
    "low_vertical_bounce": {"vertical_bounce": 0.8},
    "low_lateral_drift": {"lateral_drift": 0.8},
    "orientation_stability":        {"orientation": 0.5},   # total 0.8
    "orientation_stability_strong": {"orientation": 0.9},   # total 1.2
    "orientation_stability_mild":   {"orientation": 0.3},   # total 0.6
    "pitch_control": {"pitch_rate": 0.8, "orientation": 0.4},
    "low_roll_pitch_rate": {"roll_rate": 0.6, "pitch_rate": 0.6},
    "low_roll_rate": {"roll_rate": 0.8},
    "low_yaw_rate": {"yaw_rate": 0.6},
    "low_done_rate": {"survival": 1.0},
    "foot_clearance": {"clearance": 0.6},
    "low_scuffing": {"clearance": 0.15},
    "low_energy": {"energy": 0.6},
}
```

### 3.6 改动的文件

| 文件 | 改动 |
|------|------|
| `scripts/train_high_level_oracle_ppo.py` | 更新 `TASK_REWARD_FOCUS_WEIGHTS`：拆 token + 调权重 |
| `logs/.../training_task_map_by_speed.csv` | 更新所有行 `reward_focus` 列，删除无效 token |

---

## 4. 当前进展

### 4.1 已完成

| 任务 | 状态 |
|------|:----:|
| RMA 代码骨架（TerrainEncoder + AdaptationModule + α 退火） | ✅ |
| `--z-dim 0` 支持（no-z baseline） | ✅ |
| `--no-oracle-condition-obs` 支持（去 task_onehot） | ✅ |
| 20 iter RMA smoke test（不崩、不 NaN、adapt_loss 下降） | ✅ |
| v8 数据 per-metric gait gap 分析 | ✅ |
| Reward v4 权重 + CSV 更新 | ✅ |
| contact_style 等无效 token 清理 | ✅ |

### 4.2 待跑实验

```bash
conda activate go2_wtw

# 主实验：RMA + no-task + reward v4
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_high_level_oracle_ppo.py \
  --run-name 20260610_rma_notask_reward_v4 \
  --iterations 100 \
  --save-interval 50 \
  --adaptation-coef 0.1 \
  --no-oracle-condition-obs
```

跑完后还需要消融对比：

```bash
# B 组: RMA + task_onehot + reward v4（验证 oracle 条件下的 reward）
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_high_level_oracle_ppo.py \
  --run-name 20260610_rma_task_reward_v4 \
  --iterations 100 --save-interval 50 --adaptation-coef 0.1

# A 组: no-z + no-task + reward v4（验证 reward v4 本身的效果，无 RMA）
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_high_level_oracle_ppo.py \
  --run-name 20260610_noz_notask_reward_v4 \
  --iterations 100 --save-interval 50 --z-dim 0 --no-oracle-condition-obs
```

### 4.3 判断标准

rma_notask_reward_v4 重点看：

```
pace 是否不再坍缩（flat/ramp/rough 的 trot ratio 是否回升）
push_lateral 的 pace ratio 是否保持较高
stepping_stones 的 bound ratio 是否上升
rough_slope 不强求明显分化（v8 gap 数据本身就很小）
vx_err 不要明显恶化
clearance 不要异常暴涨（避免回到 v0 抬脚刷分）
action_clip_rate 接近 0
adaptation_loss 下降
z_error 下降
```

### 4.4 v4 实验预期

如果 v4 有效：
- flat/ramp: trot > 40%，pace < 30%
- push: pace 保持 > 35%
- stones: bound > 30%

如果仍然不分化：
- 可能需要 `soft gait prior`（selector KL to v8 template-score distribution）
- 或者需要更激进的 slip 权重（>1.2）
- rough_slope 可能天然就不分化，接受它

---

## 5. 环境信息

```text
conda env: go2_wtw
Python: 3.8
IsaacGym: /home/lekangwan/isaacgym
GPU: local RTX 4060 8GB
Project root: /home/lekangwan/run_like_a_real_dog/walk-these-ways-go2-main
```

运行时注意：
- IsaacGym 要求 `import isaacgym` 在 `import torch` 之前
- 训练脚本的 `CUDA_VISIBLE_DEVICES=0` 必须设置
- 256 env × 32 steps × 100 iter 在 RTX 4060 上约需 10-15 分钟
- 训练产出在 `runs/high_level_oracle_gait/` 下

---

## 6. 关键设计决策记录

### 为什么选 RMA 而不是 VAE 或 terrain classifier
- RMA 是底层 WTW 已经验证过的机制，工程风险最低
- Teacher-Student 蒸馏直接用 MSE，不需要设计 VAE 解码器或分类标签
- 推理时只依赖本体感知历史，可部署到真机

### 为什么不直接加 style reward
- Style reward = 查表（地形 ID → 步态），新地形无法泛化
- RMA z 是连续物理属性表征，支持插值和泛化

### 为什么 rough_slope 不强求分化
- v8 数据显示 rough_slope 上所有 trot-pace gap 都 < 0.02
- 可能 trot 和 pace 在粗糙坡道上客观上性能接近
- 如果 v4 实验证实这点，接受它

### pitch_rate 为什么不用
- v8 数据：pitch_rate 在所有地形上 trot vs pace gap ≈ 0.000
- 完全无区分度的指标不能用于反 pace

### done_rate 为什么不用
- v8 数据：所有步态的 done_rate 都在 0.004-0.007，无差异
- survival 权重 3.0 是浪费

---

## 7. sink 已有资产

```
v8 评估数据（source of truth）:
  logs/gait_condition_eval_v8_mainline/template_eval_results.csv
  logs/gait_condition_eval_v8_mainline/training_task_map/training_task_map_by_speed.csv
  logs/gait_condition_eval_v8_mainline/gait_template_library/gait_template_library.csv

当前活跃训练检查点:
  runs/high_level_oracle_gait/20260610_rma_smoke20/  (20 iter smoke test)

当前活跃 mainline 脚本:
  scripts/train_high_level_oracle_ppo.py  (训练入口)
  scripts/train_high_level_ppo.py         (ActorCritic / RolloutBuffer / 工具函数)
  scripts/gait_project_config.py          (路径和默认参数)
  scripts/gait_conditions.py              (地形条件定义)
```

---

## 8. 需警惕的坑

1. **RMA 不是银弹**：z 蒸馏成功 ≠ 步态会自然分化。需要 reward 提供足够的差异化信号。
2. **progress 主导问题**：虽然 v4 把 progress 从 2.0 降到 1.0，但如果 pace 的 progress gap 仍然显著，单靠 slip 1.2 可能不够。
3. **stepping_stones 的 bound 数据少**：v8 评估中 pronking 在 stones 上没有数据，bound vs trot 的 gap 可能不稳定。
4. **privileged_obs 扰动标志**：来自 `push_axis_by_env` 配置而非物理传感器。是 RMA 标准做法但需在文档中标明。
5. **不能长期依赖 task_onehot**：最终目标是移除它，所有消融实验必须包含 `--no-oracle-condition-obs` 版本。
6. **不要用 `--adaptation-coef 0` 冒充 no-z baseline**：z 仍然 concat 但随机/无意义。要用 `--z-dim 0`。

---

## 9. 下一步操作（按优先级）

```text
[ ] 1. 跑 rma_notask_reward_v4 (100 iter)
       → 分析 gait ratio, vx_err, clearance, adaptation_loss
[ ] 2. 如果 v4 有效 → 跑 A/B 消融，确认 reward vs RMA 分别的贡献
[ ] 3. 如果 v4 仍不分化 → 讨论 soft gait prior
[ ] 4. 消融通过后 → 跑完整的 Phase 1/2/3 三阶段训练（200-500 iter）
[ ] 5. 最终 → 移除 task_onehot 的纯 proprioception 版本，准备部署
```
