"""高层策略训练入口。

使用方法：第一阶段 ``--stage gait`` 固定连续参数，只学习步态和学生条件表征；
第二阶段 ``--stage parameters`` 从第一阶段检查点开始，冻结步态信息通路后微调连续参数。
"""

import argparse
import csv
import json
import random

import isaacgym

# Isaac Gym 必须先于 PyTorch 导入，才能正确加载其二进制扩展。
import numpy as np
import torch

from .config import (
    DEFAULT_DECISION_INTERVAL,
    DEFAULT_TASKS,
    LOW_LEVEL_LABEL,
    RUNS_DIR,
    validate_decision_interval,
)
from .environment import HighLevelEnvironment
from .low_level import find_run
from .model import HighLevelPolicy
from .ppo import PPO, RolloutBuffer
from .tasks import task_specs


def parse_args():
    """解析最小主线训练参数。

    输入：进程命令行参数。
    输出：``argparse.Namespace``，包含运行名、种子、迭代数、环境数、轨迹长度、
    决策周期、任务集合、训练阶段、可选初始化检查点和渲染开关。
    内部逻辑：只注册当前两阶段主线真正使用的选项，不暴露历史消融参数。
    作用：为可重复训练提供清晰入口，同时减少误配置空间。
    """
    parser = argparse.ArgumentParser(description="Minimal standalone high-level training.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-decisions", type=int, default=32)
    parser.add_argument("--decision-interval", type=int, default=DEFAULT_DECISION_INTERVAL)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--stage", choices=("gait", "parameters"), default="gait")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    """同步所有随机数生成器。

    输入：整数随机种子。输出：无显式返回。
    内部逻辑：依次设置 Python、NumPy、PyTorch CPU；CUDA 可用时设置全部 GPU。
    作用：让相同命令尽可能复现实验，便于多随机种子比较。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_fraction(iteration, total_iterations, stage):
    """计算当前迭代中教师隐变量的使用比例。

    输入：当前迭代、总迭代数和训练阶段。
    输出：``0.0`` 到 ``1.0`` 的浮点数 ``alpha``。
    内部逻辑：参数阶段始终为零；步态阶段前 25% 为 1，中间 50% 线性降到 0，最后保持 0。
    作用：先利用信息充分的教师稳定策略，再逐渐过渡到最终部署所需的学生。
    """
    if stage == "parameters":
        return 0.0
    progress = iteration / max(1, total_iterations - 1)
    if progress < 0.25:
        return 1.0
    if progress < 0.75:
        return 1.0 - (progress - 0.25) / 0.5
    return 0.0


def save_checkpoint(path, model, optimizer, iteration, run_config):
    """保存可恢复训练与评测的检查点。

    输入：输出路径、模型、优化器、当前迭代和运行配置字典。
    输出：无显式返回；在磁盘写入一个 ``.pt`` 文件。
    内部逻辑：确保父目录存在，再保存模型/优化器状态、迭代号和最小配置。
    作用：支持第二阶段初始化、独立评测和中断后的状态追溯。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "minimal_config": run_config,
        },
        path,
    )


def append_metrics(path, row):
    """逐迭代追加训练指标。

    输入：CSV 路径和键值结构一致的一行指标字典。
    输出：无显式返回；在磁盘创建或追加 CSV。
    内部逻辑：依据文件是否存在决定是否写表头，再按字典键顺序写入。
    作用：记录奖励、误差、损失和步态比例，供训练后分析而不依赖终端日志。
    """
    write_header = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_rollout(env, model, args, alpha):
    """采集一批在线轨迹，并把持续执行期间的奖励归于同一个步态决策。

    输入：
        env: ``HighLevelEnvironment``。
        model: 当前 ``HighLevelPolicy``。
        args: 包含轨迹决策数、保持周期和训练阶段。
        alpha: 教师隐变量混合比例。
    输出：
        buffer: 已计算回报与优势的 ``RolloutBuffer``。
        metrics: 平均奖励、终止率、速度误差和四种步态比例。
    内部逻辑：
        每个决策点采集教师/学生隐变量混合状态并采样动作；同一动作连续执行
        ``decision_interval`` 个 0.1 秒步，将折扣奖励累积到该动作；轨迹末端估计价值，
        按持续周期修正折扣因子并计算 GAE。
    作用：
        让 PPO 学习“某步态持续一段时间后的效果”，而非只看切换瞬间的噪声。
    """
    device = env.device
    selector_only = args.stage == "gait"
    augmented_dim = env.policy_obs_dim + model.latent_dim
    buffer = RolloutBuffer(
        args.rollout_decisions,
        env.num_envs,
        augmented_dim,
        env.action_dim,
        14,
        device,
    )
    observation = env.current_observation
    reward_total = 0.0
    done_total = 0.0
    vx_error_total = 0.0
    gait_counts = torch.zeros(env.num_gaits, device=device)

    for _ in range(args.rollout_decisions):
        # 教师只参与训练；随着 alpha 降低，策略逐步切换为学生隐变量。
        privileged = env.privileged_observation()
        history = observation[:, : env.base_obs_dim]
        with torch.inference_mode():
            teacher = model.encode_teacher(privileged)
            student = model.encode_student(history)
            latent = alpha * teacher + (1.0 - alpha) * student
            augmented = model.augment_observation(observation, latent)
            action, log_prob, value = model.act(augmented, selector_only)

        option_reward = torch.zeros(env.num_envs, device=device)
        option_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        active = torch.ones_like(option_done)
        # 同一组高层步态指令保持 decision_interval 个 0.1 秒步，减少瞬时奖励噪声。
        for substep in range(args.decision_interval):
            next_observation, reward, done, _ = env.step(action)
            option_reward += (0.99**substep) * reward * active.float()
            option_done |= done.bool() & active
            active &= ~done.bool()

            reward_total += reward.mean().item()
            done_total += done.float().mean().item()
            vx_error_total += torch.abs(env.measured_vx() - env.command_vx()).mean().item()
            observation = next_observation

        # 折扣后的整段回报与该段起点的动作绑定，形成一个 PPO 转移。
        buffer.add(
            augmented.detach(),
            privileged.detach(),
            action.detach(),
            log_prob.detach(),
            option_reward.detach(),
            option_done,
            value.detach(),
        )
        gait_ids = torch.argmax(action[:, : env.num_gaits], dim=-1)
        gait_counts += torch.bincount(gait_ids, minlength=env.num_gaits)

    with torch.inference_mode():
        privileged = env.privileged_observation()
        history = observation[:, : env.base_obs_dim]
        teacher = model.encode_teacher(privileged)
        student = model.encode_student(history)
        latent = alpha * teacher + (1.0 - alpha) * student
        last_obs = model.augment_observation(observation, latent)
        last_value = model.critic(last_obs).squeeze(-1)
    # 一个缓存步跨越多个物理高层步，因此折扣因子也要提升相同次幂。
    buffer.finish(last_value, 0.99**args.decision_interval, 0.95)
    env.current_observation = observation

    physical_steps = args.rollout_decisions * args.decision_interval
    return buffer, {
        "reward": reward_total / physical_steps,
        "done_rate": done_total / physical_steps,
        "vx_error": vx_error_total / physical_steps,
        "gait_ratios": gait_counts / gait_counts.sum(),
    }


def main():
    """组装环境、模型和 PPO，执行指定阶段训练并保存结果。

    输入：来自 ``parse_args`` 的命令行参数，以及内置任务目录、底层检查点等仓库资源。
    输出：无 Python 返回值；生成运行配置、训练指标 CSV 和最终模型检查点。
    内部逻辑：
        验证周期和阶段依赖，设置种子，读取任务并创建环境；构造或恢复模型，配置阶段；
        循环执行轨迹采样与 PPO 更新，记录日志，最后保存检查点。
    作用：高层最小实现的唯一训练入口，完整串起两阶段主线。
    """
    args = parse_args()
    validate_decision_interval(args.decision_interval)
    if args.stage == "parameters" and not args.init_checkpoint:
        raise ValueError("Parameter tuning requires --init-checkpoint from the gait stage.")
    set_seed(args.seed)

    task_ids = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    specs = task_specs(task_ids)
    low_level_run = find_run(LOW_LEVEL_LABEL)
    env = HighLevelEnvironment(specs, low_level_run, args.num_envs, render=args.render)
    env.current_observation = env.reset()

    model = HighLevelPolicy(
        policy_obs_dim=env.policy_obs_dim,
        base_obs_dim=env.base_obs_dim,
        num_gaits=env.num_gaits,
        residual_dim=env.action_dim - env.num_gaits,
    ).to(env.device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=env.device)
        model.load_state_dict(checkpoint["model"])
    # 阶段配置决定连续步态参数是否开放，以及哪些子网络允许更新。
    model.set_stage(args.stage)
    trainer = PPO(model, args.stage)

    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "task_ids": list(task_ids),
        "low_level_run": str(low_level_run),
        "policy_obs_dim": env.policy_obs_dim,
        "base_obs_dim": env.base_obs_dim,
        "action_dim": env.action_dim,
        "latent_dim": model.latent_dim,
    }
    with open(run_dir / "args_minimal.json", "w") as file:
        json.dump(run_config, file, indent=2)

    print(f"Run directory: {run_dir}")
    print(
        f"stage={args.stage}, decision_period={args.decision_interval * 0.1:.1f}s, "
        f"tasks={','.join(task_ids)}"
    )
    for iteration in range(args.iterations):
        alpha = teacher_fraction(iteration, args.iterations, args.stage)
        buffer, rollout_metrics = collect_rollout(env, model, args, alpha)
        losses = trainer.update(buffer)
        ratios = rollout_metrics.pop("gait_ratios")
        row = {
            "iteration": iteration,
            "teacher_fraction": alpha,
            **rollout_metrics,
            **losses,
            "pronk_ratio": ratios[0].item(),
            "trot_ratio": ratios[1].item(),
            "bound_ratio": ratios[2].item(),
            "pace_ratio": ratios[3].item(),
        }
        append_metrics(run_dir / "metrics_minimal.csv", row)
        print(
            f"iter={iteration:03d} reward={row['reward']:.3f} "
            f"vx_err={row['vx_error']:.3f} "
            f"gaits=[{row['pronk_ratio']:.2f},{row['trot_ratio']:.2f},"
            f"{row['bound_ratio']:.2f},{row['pace_ratio']:.2f}]"
        )

    save_checkpoint(
        run_dir / "checkpoints/high_level_final.pt",
        model,
        trainer.optimizer,
        args.iterations - 1,
        run_config,
    )


if __name__ == "__main__":
    main()
