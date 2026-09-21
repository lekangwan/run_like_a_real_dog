# 项目归档与复现说明

归档日期：2026-08-31

项目当前停止继续训练和结构迭代。本文件用于保证源码、最终模型、实验依据和复现边界
仍然能够被理解。最新科学结论见 `CURRENT_PROJECT_STATUS.md`。

## 1. 归档状态

```text
源码状态：保留真实 scripts 实验链和 high_level_minimal 教学实现
实验状态：停止继续扩展；底层快速切换微调尚未完成有效评测
最终高层模型：七月平地/上坡选择器模型
实机状态：未部署
运行验证：本次归档未启动 Isaac Gym，以免干扰正在运行的其他项目
```

“停止继续”不代表已经证明方法成功。当前成果包括两层控制框架、统一物理指标评测链、
局部上坡结果以及对 WTW 快速切换限制的负面诊断。不能对外声称全地形稳定泛化。

## 2. 必须保留的本地模型

### 高层汇报模型

```text
runs/high_level_oracle_gait/
20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct/
checkpoints/high_level_000049.pt
```

- 大小：6,218,962 字节
- SHA-256：`f45304896cd341dc0f5a97a6386833925f89f8be40045f91c4467791ef02bc23`

### 原始 WTW 底层运行

```text
runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/
```

必须一起保存：

| 文件 | 大小（字节） | SHA-256 |
|---|---:|---|
| `parameters.pkl` | 236,544 | `8b6a39e9f86332d044c5d0f5a3794667cb671b8f44fdb70c6a95904f0588ea12` |
| `checkpoints/ac_weights_019999.pt` | 12,225,632 | `1f4218009a9d269ffb54b9034b6a488b09062fda6d1115cd4ac7943a70a81c43` |
| `checkpoints/body_latest.jit` | 4,981,032 | `7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5` |
| `checkpoints/adaptation_module_latest.jit` | 2,293,757 | `0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13` |

底层微调只完成过修复配置恢复后的 5 次更新检查，不是已验证的新底层模型。项目归档
时已删除全部未验证的底层微调检查点和无效评测，只保留
`scripts/finetune_low_level_gait_transitions.py`。恢复项目时必须从原 WTW 最终模型重新开始。

仓库根目录的 `MODEL_ARTIFACTS.sha256` 保存了上述文件的机器可读校验值。在模型目录
仍存在时，可以执行以下纯文件检查：

```bash
sha256sum -c MODEL_ARTIFACTS.sha256
```

## 3. Git 仓库不能单独完整复现

`.gitignore` 排除了：

```text
runs/
logs/
*.pt、*.jit 等模型文件
```

因此，从 GitHub 新克隆仓库只能获得源码、文档和部分精选汇报素材，不能直接复现最终
模型。要保留复现可能性，必须另行备份上一节的模型目录以及需要引用的原始评测结果。

Isaac Gym Preview 4 也不是普通 `pip` 依赖，必须由使用者单独安装，并保证 Python、
PyTorch、CUDA 和显卡驱动兼容。

## 4. 已知运行环境

项目实际使用过：

```text
Ubuntu
Python 3.8
conda 环境：go2_wtw
Isaac Gym Preview 4
PyTorch 2.4.1 + CUDA 12.1
NumPy 1.23.5
Gym 旧版接口
PhysX GPU 或 CPU 管线，取决于具体实验
```

`setup.py` 只声明基础 Python 依赖，不是完整环境锁定文件。恢复环境时应先安装 Isaac
Gym，再安装匹配的 PyTorch，最后在仓库根目录执行：

```bash
python3 -m pip install -e .
```

仅检查导入，不启动仿真：

```bash
python3 -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
python3 -c "import isaacgym; print('Isaac Gym import OK')"
```

注意：Isaac Gym 通常要求在同一进程中先导入 `isaacgym`，再导入 `torch`。

## 5. 真实代码最小保留链

最终结果不是由 `high_level_minimal/` 生成，而是由以下 `scripts/` 文件共同产生：

```text
gait_project_config.py
gait_conditions.py
train_high_level_ppo.py
train_high_level_oracle_ppo.py
evaluate_high_level_policy_by_task.py
analyze_adaptive_vs_forced_trot.py
```

为了理解奖励、公平审查和八月修正结论，还应保留：

```text
evaluate_paired_gait_live_reward.py
evaluate_gait_target_fairness.py
check_high_level_reward_consistency.py
analyze_wtw_capability_frontier.py
finetune_low_level_gait_transitions.py
```

这些脚本使用同目录直接导入，不能在不修改导入方式的情况下任意移动。完整分类见
`scripts/README.md`。

## 6. 原则上的最终模型评测命令

下面的命令会启动 Isaac Gym，需要独占或可接受的计算资源。归档时没有重新执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/scripts:$PWD python3 -B \
  scripts/evaluate_high_level_policy_by_task.py \
  --run-dir runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct \
  --checkpoint runs/high_level_oracle_gait/20260720_v4_flat_ramp_per_env_baseline_seed22550_stage2_iter050_direct/checkpoints/high_level_000049.pt \
  --eval flat_trot_efficiency:0.5,flat_trot_efficiency:1.0,flat_trot_efficiency:1.5,flat_trot_efficiency:2.0,ramp_up_trot_robustness:0.5,ramp_up_trot_robustness:1.0,ramp_up_trot_robustness:1.5,ramp_up_trot_robustness:2.0 \
  --num-envs 32 \
  --steps 1000 \
  --output-dir runs/high_level_oracle_gait/archive_recheck
```

评测自适应策略时还必须运行相同条件下的强制小跑版本，随后使用
`analyze_adaptive_vs_forced_trot.py` 汇总。只运行自适应策略不能证明它有价值。

## 7. 不能被误认为最终结果的内容

- `high_level_minimal/`：教学版重写，没有完成 Isaac Gym 等价复现；
- 显式步态参考训练：只用于证明选择器可学习，包含训练先验；
- 第一版底层微调配置恢复错误，结果已作废且原始目录已清理；
- 单个随机种子或单次视觉演示：不能证明性能改善；
- 七月 10 秒步态保持模型：不能证明快速地形适应；
- PPT 中强制指定步态的视频：只能展示低层动作能力，不能当成策略自主选择证据。

## 8. 最终阅读顺序

1. `README.md`：项目概览；
2. `CURRENT_PROJECT_STATUS.md`：最终可信结论和失败边界；
3. `ARCHIVE_AND_REPRODUCIBILITY.md`：模型与复现条件；
4. `scripts/README.md`：真实代码入口；
5. `docs/history/DETAILED_PROJECT_REVIEW_20260723.md`：七月以前的技术演化；
6. `docs/history/CURRENT_GAIT_ADAPTATION_PLAN.md`：需要追查具体实验时再检索。

## 9. 归档后的维护原则

- 不再向大型实验日志持续追加普通说明；
- 不删除最终模型和原 WTW 源运行配置；
- 不把教学版代码描述为最终实验实现；
- 不通过移动 `scripts/` 文件来追求表面整洁；
- 若未来恢复项目，先验证底层快速转换，再决定是否重新训练短周期高层。
