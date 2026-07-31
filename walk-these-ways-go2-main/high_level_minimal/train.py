import argparse
import csv
import json
import random

import isaacgym

assert isaacgym
import numpy as np
import torch

from .config import (
    DEFAULT_DECISION_INTERVAL,
    DEFAULT_TASKS,
    LOW_LEVEL_LABEL,
    RUNS_DIR,
    TASK_MAP,
    validate_decision_interval,
)
from .environment import HighLevelEnvironment
from .low_level import find_run
from .model import HighLevelPolicy
from .ppo import PPO, RolloutBuffer
from .tasks import read_task_specs


def parse_args():
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_fraction(iteration, total_iterations, stage):
    if stage == "parameters":
        return 0.0
    progress = iteration / max(1, total_iterations - 1)
    if progress < 0.25:
        return 1.0
    if progress < 0.75:
        return 1.0 - (progress - 0.25) / 0.5
    return 0.0


def save_checkpoint(path, model, optimizer, iteration, run_config):
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
    write_header = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_rollout(env, model, args, alpha):
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
        for substep in range(args.decision_interval):
            next_observation, reward, done, _ = env.step(action)
            option_reward += (0.99**substep) * reward * active.float()
            option_done |= done.bool() & active
            active &= ~done.bool()

            reward_total += reward.mean().item()
            done_total += done.float().mean().item()
            vx_error_total += torch.abs(env.measured_vx() - env.command_vx()).mean().item()
            observation = next_observation

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
    args = parse_args()
    validate_decision_interval(args.decision_interval)
    if args.stage == "parameters" and not args.init_checkpoint:
        raise ValueError("Parameter tuning requires --init-checkpoint from the gait stage.")
    set_seed(args.seed)

    task_ids = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    specs = read_task_specs(TASK_MAP, task_ids)
    low_level_run = find_run(LOW_LEVEL_LABEL)
    env = HighLevelEnvironment(specs, low_level_run, args.num_envs, render=args.render)
    env.current_observation = env.reset()

    model = HighLevelPolicy(
        policy_obs_dim=env.policy_obs_dim,
        base_obs_dim=env.base_obs_dim,
        num_gaits=env.num_gaits,
        residual_dim=env.action_dim - env.num_gaits,
        use_gait_input_residuals=args.stage == "parameters",
    ).to(env.device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=env.device)
        model.load_state_dict(checkpoint["model"], strict=False)
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
