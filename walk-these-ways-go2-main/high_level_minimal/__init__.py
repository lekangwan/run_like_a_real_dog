"""Go2 高层步态自适应主线的可阅读最小实现。

模块作用：
    把散落在历史 ``scripts/`` 中的高层主线整理为一个可独立导入的 Python 包。

外部输入：
    原仓库的 ``go2_gym`` 基础设施、Isaac Gym 和冻结 WTW 检查点。

内部处理：
    由训练、环境、步态包装器、师生模型、PPO、评测和录像模块协作完成高层控制。

对外输出：
    可通过 ``python -m high_level_minimal.train/evaluate/record`` 直接调用的完整入口。

边界：
    本包不复制底层仿真和 WTW 实现，但不依赖原 ``scripts/`` 中的任何 Python 文件。
"""
