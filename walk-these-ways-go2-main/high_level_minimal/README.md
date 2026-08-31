# Go2 高层步态控制：教学最小实现

本目录用于学习项目结构，不是历史实验脚本的替代入口。它把主线方案整理成一条
单一、可阅读的实现路径，并且不导入 `scripts/`。即使移走 `scripts/`，本目录在
接口上仍可完成训练、评测和录像，但只能读取**由本目录自己生成的高层检查点**。

它仍依赖以下基础设施，这些内容不属于高层整理范围：

- `go2_gym/`：Go2、Isaac Gym 环境和底层历史包装器；
- `resources/`：机器人和地形资源；
- `runs/gait-conditioned-agility/`：冻结的 WTW 底层模型；
- Isaac Gym、PyTorch、NumPy 和 OpenCV。

真实项目结果来自原 `scripts/`。本目录是从这些实验中提炼出的教学基线，尚未经过
完整 Isaac Gym 短训练验证，不能把它与历史最终检查点混用。

## 一、项目要验证什么

```text
只使用目标速度和本体感知历史，
高层策略能否选择步态并小幅修正连续步态参数，
在统一物理评价下优于固定 trot。
```

“不同地形必须出现不同步态”不是成功条件。若策略最终几乎总选 trot，就必须靠
连续参数或局部切换带来可重复的性能提升，否则高层没有实际价值。

## 二、完整数据流

```text
目标速度 + 10 帧本体历史（511 维）
              |
              v
学生编码器：510 维历史 -> 16 维条件表征
              |
              +----> 预测 14 维通用物理量
              |
              v
步态选择器：目标速度 + 表征 + 预测物理量 -> 4 种步态概率
              |
              v
连续参数网络：完整状态 + 已选步态 -> 5 个参数修正量
              |
              v
9 维高层步态指令：4 维步态 + 5 维参数修正
              |
              v
冻结 WTW 底层策略 -> 12 维关节动作 -> PD 控制与物理仿真
```

这里的 9 维输出是“高层步态指令”，不是电机动作。PPO 代码沿用算法术语
`action`，指的仍是这 9 维指令；真正的底层动作是 WTW 输出的 12 维关节目标。

训练时教师把 14 维仿真物理量编码成 16 维表征，学生用本体历史模仿它。部署和
独立评测只使用学生，不读取任务编号、地形标签、相机或雷达。

## 三、文件职责与阅读顺序

1. `config.py`：路径、步态参数名称和四层时间尺度。
2. `tasks.py`：五类场景、速度范围和并行环境分配。
3. `low_level.py`：恢复冻结 WTW 并创建 Isaac Gym 环境。
4. `reward.py`：九项统一物理评分及固定权重。
5. `gait_wrapper.py`：把 9 维指令映射为 WTW 命令，并构造本体历史。
6. `environment.py`：组合任务、底层策略和高层包装器。
7. `model.py`：教师、学生、步态选择器、连续参数网络和价值网络。
8. `ppo.py`：在线轨迹缓存、优势估计和 PPO 更新。
9. `train.py`：两阶段训练入口。
10. `evaluate.py`：自适应策略与固定步态的同口径评测。
11. `record.py`、`visualize.py`：录像和可视化。

更细的阅读问题见 `LEARNING_ROADMAP.md`。代码注释以函数为单位说明输入、输出、
处理逻辑和作用；没有给每一行重复写注释。

## 四、时间尺度

```text
物理仿真：             0.005 秒，200 Hz
WTW 底层策略：         0.020 秒， 50 Hz
高层环境：             0.100 秒， 10 Hz
默认重新选择步态：     0.500 秒，  2 Hz
本体历史窗口：         1.000 秒，10 个高层帧
```

高层决策保持范围被限制为 1 至 10 个高层步，即 0.1 至 1.0 秒。历史实验中的
100 步等于 10 秒，不符合在线适应需求，因此教学版直接拒绝该设置。

## 五、统一奖励

`reward.py` 只保留当前教学方案实际使用的九项：

- 核心任务：前进速度跟踪、偏航跟踪、存活；
- 基础稳定：身体姿态；
- 接触安全：侧漂、接触滑移、落地冲击、擦碰；
- 效率：机械功率，但只在速度跟踪较好时开启。

所有地形使用同一公式和权重。统一规则能减少地形标签先验，但不自动保证物理公平；
阈值、归一化尺度和 WTW 本身的能力仍需通过固定步态对照与原始物理量审查。

## 六、两阶段训练

第一阶段固定连续参数，只学习条件表征和步态选择：

```bash
python3 -m high_level_minimal.train \
  --run-name minimal_gait_stage \
  --stage gait \
  --decision-interval 5 \
  --iterations 50
```

第二阶段从第一阶段开始，冻结步态信息通路，只微调连续参数和价值网络：

```bash
python3 -m high_level_minimal.train \
  --run-name minimal_parameter_stage \
  --stage parameters \
  --init-checkpoint runs/high_level_oracle_gait/minimal_gait_stage/checkpoints/high_level_final.pt \
  --decision-interval 5 \
  --iterations 30
```

连续参数网络的输出层从零开始，探索标准差降为 0.1，并用零点正则限制其只在
默认 WTW 模板附近调节。这一安排隔离了“选错步态”和“参数扰坏”两类奖励来源。

## 七、独立评测

```bash
python3 -m high_level_minimal.evaluate \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval flat_trot_efficiency:1.0,ramp_up_trot_robustness:1.0
```

固定 trot 对照只替换高层指令，环境和统计口径不变：

```bash
python3 -m high_level_minimal.evaluate \
  --run-dir runs/high_level_oracle_gait/minimal_gait_stage \
  --eval flat_trot_efficiency:1.0,ramp_up_trot_robustness:1.0 \
  --force-gait trotting
```

## 八、刻意删除的内容

教学版不包含任务编号输入、目标步态奖励、步态参考分布、逐地形奖励、10 秒动作锁定、
公平网格搜索、离线重打分、信息通路探针和旧检查点兼容。这些属于历史实验或诊断，
会掩盖主线数据流。

仍保留的少量检查有明确用途：参数范围裁剪避免非法物理命令；录像器检查相机和编码器；
`low_level.py` 只在恢复 WTW 的旧 `parameters.pkl` 时检查字段是否存在。后者是唯一明显的
历史兼容边界，集中在一个文件中，没有扩散到模型、训练或评测逻辑。

## 九、验证状态

本次整理已完成：

- 所有 Python 文件的语法解析；
- 确认没有导入 `scripts/`；
- CPU 上的任务分配、教师/学生维度、两阶段指令和奖励数值检查；
- 检查不存在 `strict=False`、旧配置回退和多套连续参数头。

尚未启动 Isaac Gym，也未运行短训练、独立评测或录像。这是为了避免影响当前正在运行的
其他仿真项目。因此目前可以称为“静态一致、适合阅读的教学实现”，还不能称为经过端到端
运行验证的新训练主线。
