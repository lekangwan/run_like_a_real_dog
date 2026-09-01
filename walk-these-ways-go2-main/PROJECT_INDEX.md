# Go2 项目文件总索引

更新日期：2026-08-31

这份文件只回答两个问题：现在应该从哪里开始看，以及不同文档分别有什么用途。

## 1. 日常入口

1. [`README.md`](README.md)：项目目标、结构、主要结果和运行入口。
2. [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md)：当前唯一状态入口、证据边界和下一阶段顺序。
3. [`ARCHIVE_AND_REPRODUCIBILITY.md`](ARCHIVE_AND_REPRODUCIBILITY.md)：归档状态、模型清单与复现边界。
4. [`scripts/README.md`](scripts/README.md)：真实实验代码的分类与阅读顺序。
5. [`INTERVIEW_QA.md`](INTERVIEW_QA.md)：面试问题和口述答案。

`PROJECT_STATUS_20260723.md` 和 `REPORT_READY_PROJECT_STATUS_20260721.md` 都是七月汇报快照，
不包含八月对 WTW 能力边界和快速切换瞬态的修正结论。

## 2. 深入理解

| 文件 | 用途 |
|---|---|
| `CURRENT_PROJECT_STATUS.md` | 截至 2026-08-31 的可信结论、未完成项和下一步 |
| `ARCHIVE_AND_REPRODUCIBILITY.md` | 本地模型、环境依赖、可复现与不可复现边界 |
| `MODEL_ARTIFACTS.sha256` | 最终高层模型、原 WTW 模型与源配置的文件校验值 |
| `DETAILED_PROJECT_REVIEW_20260723.md` | 重建项目目标、结构、奖励演变和关键实验逻辑 |
| `REPORT_READY_PROJECT_STATUS_20260721.md` | 汇报数据、图表和结论边界 |
| `high_level_minimal/LEARNING_ROADMAP.md` | 从零阅读高层控制代码的学习顺序 |
| `high_level_minimal/README.md` | 教学版最小实现说明 |

`high_level_minimal/` 用于学习结构，不是当前继续实验的开发基础；真实训练和评测仍使用
`scripts/` 中的代码。

## 3. 历史流水账

| 文件 | 性质 |
|---|---|
| `ACTIVE_PROJECT_CONTEXT.md` | 早期至中期的完整技术上下文，体积较大 |
| `CURRENT_GAIT_ADAPTATION_PLAN.md` | 后续逐轮实验记录和命令，体积最大 |
| `CONVERSATION_HANDOFF_*.md` | 早期对话交接快照 |

这些文件用于追溯，不应作为第一次阅读入口，也不应继续无限追加作为当前状态摘要。

## 4. 代码边界

```text
scripts/             当前真实高层训练、评测和历史实验代码
high_level_minimal/  用于学习的独立高层最小版本
go2_gym/             仿真环境、机器人、奖励和高层包装器
go2_gym_learn/       原 WTW 强化学习基础设施
go2_gym_deploy/      Unitree 部署基础设施
```

## 5. 数据与汇报

```text
runs/       本地训练与评测结果，不作为源码阅读入口
logs/       底层模型和历史扫描日志
reports/    精选阶段汇报材料
ppt_assets/ 面试 PPT 图片、视频和生成脚本
```

## 6. 当前整理边界

- 不删除底层 WTW 代码；
- 不删除 `runs/` 和 `logs/` 中的历史结果；
- 不把教学版最小实现替换为真实实验主线；
- 暂不移动 `scripts/` 文件，先通过索引分类，避免破坏历史导入和命令；
- 后续新增面试问题统一更新 `INTERVIEW_QA.md`。
