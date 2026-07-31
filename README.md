# 基于本体感知历史的 Go2 高层步态自适应

本仓库研究如何在冻结的 Walk These Ways（WTW）底层运动策略之上，训练一个高层
策略，仅根据目标速度和机器人本体感知历史选择步态，并尝试调整连续步态参数。

当前项目代码位于：

[`walk-these-ways-go2-main/`](walk-these-ways-go2-main/)

建议从以下入口开始：

- [项目完整说明](walk-these-ways-go2-main/README.md)
- [最小高层实现](walk-these-ways-go2-main/high_level_minimal/)
- [从零学习路线](walk-these-ways-go2-main/high_level_minimal/LEARNING_ROADMAP.md)
- [当前结果与局限](walk-these-ways-go2-main/PROJECT_STATUS_20260723.md)
- [详细技术复盘](walk-these-ways-go2-main/DETAILED_PROJECT_REVIEW_20260723.md)

本项目基于
[walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
以及其 Unitree Go2 移植版本继续开发。上游代码、第三方组件及其许可证归原作者所有。
