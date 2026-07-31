# 高层步态主线：可独立运行的最小代码

本目录包含当前高层方案所需的完整代码。它不导入 `scripts/` 中的任何文件。
即使移走原 `scripts/`，这里仍可完成高层训练、评测、可视化和录像。

从零开始学习本项目时，请先阅读 `LEARNING_ROADMAP.md`。它记录了固定阅读顺序、
每个文件需要回答的问题、动手练习和掌握主线后的自检标准。

它只依赖不在本次整理范围内的基础设施：

- `go2_gym/`：机器人环境、奖励和高低层封装；
- `resources/`：机器人与地形资源；
- `runs/gait-conditioned-agility/`：已经训练好的 WTW 低层模型；
- Isaac Gym、PyTorch、NumPy、OpenCV。

## 阅读顺序

建议依次阅读：

1. `config.py`：项目路径、时间尺度和默认设置；
2. `tasks.py`：怎样读取任务、分配并行环境、设置统一奖励；
3. `low_level.py`：怎样加载冻结的 WTW 低层模型；
4. `gait_wrapper.py`：步态与连续参数怎样变成 WTW 命令，奖励怎样计算；
5. `environment.py`：怎样把任务、底层环境和步态封装器连接起来；
6. `model.py`：步态选择、连续参数、教师和学生网络；
7. `ppo.py`：强化学习数据与更新公式；
8. `train.py`：如何把前七部分组成完整训练；
9. `evaluate.py`：如何独立评测和比较固定步态；
10. `visualize.py`、`record.py`：观察和录像。

主线中的步态封装器由原文件的 864 行缩减到约 450 行，删除了目标步态标签、
参考步态惩罚、动作锁定和历史审查导出等非主线功能。其他代码文件不超过约
250 行。

## 完整信息流

```text
目标速度 + 10 帧本体感知历史
                |
                v
学生编码器：历史 -> 16 维环境表征
                |
                +----> 预测 14 维通用物理状态
                |
                v
目标速度 + 环境表征 + 预测物理状态
                |
                v
步态选择器：pronk / trot / bound / pace
                |
                v
高层步态动作 + 可选的 5 个连续参数修正
                |
                v
冻结的 WTW 低层策略
                |
                v
12 个关节动作
```

训练时，教师编码器直接读取仿真提供的 14 维物理量，为学生的 16 维表征提供
学习目标。推理时只使用学生，不使用任务编号、地形标签、相机或雷达。

## 两阶段训练

阶段一只训练步态，连续参数固定为默认值：

```bash
python3 -m high_level_minimal.train \
  --run-name minimal_gait_stage \
  --stage gait \
  --decision-interval 5 \
  --iterations 50
```

阶段二必须从阶段一检查点开始，只调连续参数与对应的价值估计：

```bash
python3 -m high_level_minimal.train \
  --run-name minimal_parameter_stage \
  --stage parameters \
  --init-checkpoint runs/high_level_oracle_gait/minimal_gait_stage/checkpoints/high_level_final.pt \
  --decision-interval 5 \
  --iterations 30
```

阶段二会：

- 冻结步态选择器；
- 冻结教师、学生和物理状态预测；
- 将连续参数输出重新初始化为零；
- 将连续参数探索标准差设为 `0.1`；
- 使用小幅偏离默认参数的惩罚。

录像按高层仿真的真实 `0.1` 秒步长推进。请求 30 帧视频时，同一个仿真画面
会按需重复写入，保证视频时长和仿真时长一致，不会把机器人动作加速三倍。

## 时间尺度保护

高层环境每步为 `0.1` 秒。新代码只允许：

```text
decision_interval = 1～10
实际步态更新周期 = 0.1～1.0 秒
```

历史实验中的 `decision_interval=100` 等于 10 秒，已被认定不适合在线地形
适应，新代码会直接拒绝该设置。

需要注意：当前实现仍在一个决策周期内保持动作不变。`0.5` 秒只是下一轮实验
的合理起点，不代表已经证明它是最终最佳周期。

## 独立评测

自适应策略：

```bash
python3 -m high_level_minimal.evaluate \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval flat_trot_efficiency:1.0,ramp_up_trot_robustness:1.0
```

固定 trot 对照：

```bash
python3 -m high_level_minimal.evaluate \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval flat_trot_efficiency:1.0,ramp_up_trot_robustness:1.0 \
  --force-gait trotting
```

打开界面：

```bash
python3 -m high_level_minimal.visualize \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval ramp_up_trot_robustness:1.0
```

录制视频：

```bash
python3 -m high_level_minimal.record \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval ramp_up_trot_robustness:1.0 \
  --output-dir reports/minimal_videos
```

## 明确删除的历史功能

这个最小版本没有包含：

- 任务编号直接输入；
- 人工指定的目标步态奖励；
- 步态参考分布监督；
- 每种地形不同的奖励；
- 旧版统一奖励候选；
- 10 秒步态锁定；
- 大量只为某一次排错增加的命令行参数；
- 公平步态网格、奖励重打分和信息通路审查工具。

这些仍保留在原 `scripts/`，用于追溯历史，但不属于当前主线。

## 当前验证状态

已经完成：

- 所有文件语法检查；
- 检查确认没有导入原 `scripts`；
- CPU 上的策略动作、价值和学生推理尺寸检查；
- 任务读取、环境分配和 36 维统一奖励权重检查。

尚未完成：

- Isaac Gym 环境启动测试；
- 5 轮短训练测试；
- 新训练检查点的独立评测；
- `0.1～1.0` 秒不同步态更新周期的正式比较。

这些涉及显卡和仿真，应在明确安排后运行。当前目录是完整实现，但在短训练验证
通过前，不能称为已验证的新主线。
