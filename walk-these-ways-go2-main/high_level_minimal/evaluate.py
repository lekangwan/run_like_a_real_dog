"""高层策略的独立批量评测入口。

同一脚本既可评测自适应策略，也可通过 ``--force-gait`` 强制固定步态，
从而在相同检查点、地形、速度和统计口径下构造公平基线。
"""

import argparse
import csv
import json
from pathlib import Path

import isaacgym

# Isaac Gym 必须先于 PyTorch 导入。
import torch

from .config import validate_decision_interval
from .environment import HighLevelEnvironment
from .model import HighLevelPolicy
from .tasks import fixed_gait_action, parse_eval_specs


def parse_args():
    """解析独立评测参数。

    输入：进程命令行。
    输出：包含运行目录、检查点、任务速度点、环境数、步数、决策周期、固定步态和输出目录。
    内部逻辑：要求明确运行目录和至少一个 ``任务:速度``，其他参数提供保守默认值。
    作用：同一入口支持自适应策略和固定步态基线，减少两套评测代码产生的口径差异。
    """
    parser = argparse.ArgumentParser(description="Evaluate the minimal high-level policy.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--eval",
        required=True,
        help="Comma-separated task:speed pairs.",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--decision-interval", type=int, default=None)
    parser.add_argument(
        "--force-gait",
        choices=("pronking", "trotting", "bounding", "pacing"),
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def latest_checkpoint(run_dir):
    """定位运行目录中的最新检查点。

    输入：``Path`` 类型运行目录。
    输出：检查点 ``Path``；目录为空时抛出 ``FileNotFoundError``。
    内部逻辑：列举 ``checkpoints/*.pt`` 并按文件名排序，返回最后一项。
    作用：允许省略 ``--checkpoint``，减少日常评测命令长度。
    """
    candidates = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {run_dir / 'checkpoints'}")
    return candidates[-1]


def load_run_config(run_dir):
    """读取教学版训练保存的运行配置。

        输入：运行目录 ``Path``。
        输出：训练参数字典。
        内部逻辑：读取本目录训练入口固定生成的 ``args_minimal.json``。
        作用：严格按训练时维度和阶段重建模型，不承担历史检查点兼容工作。
    """
    with open(run_dir / "args_minimal.json") as file:
        return json.load(file)


def create_model(env, config):
    """依据检查点配置重建网络结构，并判断是否仅选择步态。

    输入：已经创建的环境和运行配置字典。
    输出：``(model, selector_only)``，模型已移动到环境设备但尚未加载权重。
    内部逻辑：
        按环境观测/动作维度构造模型，再根据训练阶段设置连续参数掩码。
    作用：保证评测网络结构与本教学版训练检查点一致。
    """
    stage = config["stage"]
    selector_only = stage == "gait"
    model = HighLevelPolicy(
        policy_obs_dim=env.policy_obs_dim,
        base_obs_dim=env.base_obs_dim,
        num_gaits=env.num_gaits,
        residual_dim=env.action_dim - env.num_gaits,
        latent_dim=int(config["latent_dim"]),
    ).to(env.device)
    model.residual_mask.fill_(0.0 if selector_only else 1.0)
    return model, selector_only


def write_rows(path, rows):
    """将每个任务速度点的汇总结果写成 CSV。

    输入：输出路径和非空行字典列表。
    输出：无显式返回；创建父目录并覆盖写入 CSV。
    内部逻辑：使用首行键作为固定表头，再顺序写入全部结果。
    作用：为后续成对汇总、绘图和随机种子比较提供结构化数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    """执行确定性学生策略或固定步态，并按任务分别累计指标。

    输入：命令行配置、运行目录中的模型配置和检查点、内置任务目录。
    输出：无 Python 返回值；写入 ``summary.csv`` 并在终端打印每个测试点摘要。
    内部逻辑：
        恢复配置和模型，按固定速度建立并行环境；每到决策边界生成学生动作或固定动作；
        按任务掩码累计奖励、速度误差、终止和步态计数，最后按真实样本数归一化。
    作用：在完全相同的仿真与统计协议下判断自适应策略是否优于固定步态。
    """
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_run_config(run_dir)
    saved_interval = int(config.get("decision_interval", 1))
    decision_interval = args.decision_interval or saved_interval
    validate_decision_interval(decision_interval)

    specs = parse_eval_specs(args.eval)
    low_level_run = Path(config["low_level_run"])
    env = HighLevelEnvironment(
        specs,
        low_level_run,
        args.num_envs,
        render=args.render,
    )
    observation = env.reset()
    model, selector_only = create_model(env, config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    task_ids = env.assignment.task_ids
    reward_sum = torch.zeros(len(specs), device=env.device)
    vx_error_sum = torch.zeros(len(specs), device=env.device)
    done_sum = torch.zeros(len(specs), device=env.device)
    gait_counts = torch.zeros(len(specs), env.num_gaits, device=env.device)
    sample_counts = torch.zeros(len(specs), device=env.device)

    for step in range(args.steps):
        # 仅到达决策边界时更新步态，其余 0.1 秒步继续执行原动作。
        if step % decision_interval == 0:
            with torch.inference_mode():
                if args.force_gait:
                    action = fixed_gait_action(
                        env.num_envs,
                        args.force_gait,
                        env.device,
                    )
                else:
                    action = model.act_student(observation, selector_only)

        observation, reward, done, _ = env.step(action)
        gait_ids = torch.argmax(action[:, : env.num_gaits], dim=-1)
        for task_index in range(len(specs)):
            mask = task_ids == task_index
            count = mask.sum()
            reward_sum[task_index] += reward[mask].sum()
            vx_error_sum[task_index] += torch.abs(
                env.measured_vx()[mask] - env.command_vx()[mask]
            ).sum()
            done_sum[task_index] += done[mask].float().sum()
            gait_counts[task_index] += torch.bincount(
                gait_ids[mask],
                minlength=env.num_gaits,
            )
            sample_counts[task_index] += count

    # 所有累计量除以真实样本数，避免任务环境数量不同造成偏差。
    rows = []
    for task_index, spec in enumerate(specs):
        denominator = sample_counts[task_index].clamp_min(1)
        gait_ratio = gait_counts[task_index] / gait_counts[task_index].sum().clamp_min(1)
        rows.append(
            {
                "task_id": spec.task_id.split("@", 1)[0],
                "condition": spec.condition,
                "cmd_vx": spec.vx_low,
                "reward": (reward_sum[task_index] / denominator).item(),
                "vx_error": (vx_error_sum[task_index] / denominator).item(),
                "done_rate": (done_sum[task_index] / denominator).item(),
                "pronk_ratio": gait_ratio[0].item(),
                "trot_ratio": gait_ratio[1].item(),
                "bound_ratio": gait_ratio[2].item(),
                "pace_ratio": gait_ratio[3].item(),
                "forced_gait": args.force_gait or "",
                "decision_interval": decision_interval,
            }
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "independent_eval" / "minimal"
    )
    write_rows(output_dir / "summary.csv", rows)
    for row in rows:
        print(
            f"{row['task_id']} vx={row['cmd_vx']:.2f} "
            f"reward={row['reward']:.3f} vx_err={row['vx_error']:.3f} "
            f"gaits=[{row['pronk_ratio']:.2f},{row['trot_ratio']:.2f},"
            f"{row['bound_ratio']:.2f},{row['pace_ratio']:.2f}]"
        )


if __name__ == "__main__":
    main()
