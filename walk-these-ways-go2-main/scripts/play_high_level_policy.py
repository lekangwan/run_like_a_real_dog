import argparse
from pathlib import Path
import time

import isaacgym

assert isaacgym
import torch

from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper
from scripts.train_high_level_ppo import (
    ActorCritic,
    find_logdir,
    load_low_level_env,
    load_low_level_policy,
)


def parse_vx_list(value):
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--vx-list must contain at least one value")
    return values


def load_high_level_model(checkpoint_path, obs_dim, action_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ActorCritic(obs_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("iteration", -1))


def format_action_stats(env):
    mapped = env._map_action(env.high_level_action)
    selector_weights = mapped["selector_weights"].mean(dim=0)
    selector_text = " ".join(
        f"{name}={selector_weights[i].item():.2f}"
        for i, name in enumerate(env.gait_names)
    )
    return (
        f"action_mean={env.high_level_action.mean(dim=0).detach().cpu().numpy()} "
        f"{selector_text} "
        f"phase={mapped['phase'].mean().item():.3f} "
        f"offset={mapped['offset'].mean().item():.3f} "
        f"bound={mapped['bound'].mean().item():.3f} "
        f"freq={mapped['frequency'].mean().item():.3f} "
        f"footswing={mapped['footswing_height'].mean().item():.3f} "
        f"stance_width={mapped['stance_width'].mean().item():.3f} "
        f"body_pitch={mapped['body_pitch'].mean().item():.3f}"
    )


def format_per_env_action_stats(env, max_envs=8):
    mapped = env._map_action(env.high_level_action)
    selector_weights = mapped["selector_weights"]
    lines = []
    count = min(env.num_envs, max_envs)
    for i in range(count):
        selector_text = " ".join(
            f"{name}={selector_weights[i, gait_id].item():.2f}"
            for gait_id, name in enumerate(env.gait_names)
        )
        lines.append(
            "  "
            f"env={i:02d} "
            f"cmd_vx={env.commands[i, 0].item():.2f} "
            f"measured_vx={env.base_lin_vel[i, 0].item():.2f} "
            f"{selector_text} "
            f"phase={mapped['phase'][i].item():.3f} "
            f"offset={mapped['offset'][i].item():.3f} "
            f"bound={mapped['bound'][i].item():.3f} "
            f"freq={mapped['frequency'][i].item():.3f} "
            f"footswing={mapped['footswing_height'][i].item():.3f} "
            f"stance_width={mapped['stance_width'][i].item():.3f} "
            f"body_pitch={mapped['body_pitch'][i].item():.3f} "
            f"action={env.high_level_action[i].detach().cpu().numpy()}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--vx-list", type=parse_vx_list, default=None)
    parser.add_argument("--spread-vx", action="store_true")
    parser.add_argument("--switch-interval", type=int, default=500)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logdir = find_logdir(args.label, args.run_index)
    low_env = load_low_level_env(logdir, args.num_envs, render=not args.no_render)
    low_policy = load_low_level_policy(logdir)
    env = HighLevelGaitWrapper(low_env, low_policy, record_reward_terms=True)

    device = env.device
    model, checkpoint_iteration = load_high_level_model(
        checkpoint_path,
        env.num_high_level_obs_history,
        env.num_high_level_actions,
        device,
    )

    obs = env.reset()
    vx_values = args.vx_list if args.vx_list is not None else [args.vx]
    if args.spread_vx:
        vx_tensor = torch.tensor(vx_values, device=device, dtype=torch.float)
        repeats = (args.num_envs + len(vx_values) - 1) // len(vx_values)
        vx_tensor = vx_tensor.repeat(repeats)[: args.num_envs]
        env.set_velocity_command(vx_tensor, args.vy, args.yaw)
    else:
        env.set_velocity_command(vx_values[0], args.vy, args.yaw)

    reward_sum = 0.0
    done_sum = 0.0
    vx_error_sum = 0.0
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(iteration={checkpoint_iteration})"
    )
    print(
        f"num_envs={args.num_envs}, obs_dim={env.num_high_level_obs_history}, "
        f"action_dim={env.num_high_level_actions}, render={not args.no_render}"
    )
    if args.spread_vx:
        print(f"spread vx commands: {env.commands[:, 0].detach().cpu().numpy()}")

    with torch.inference_mode():
        for step in range(args.steps):
            if not args.spread_vx and len(vx_values) > 1 and step % args.switch_interval == 0:
                vx = vx_values[(step // args.switch_interval) % len(vx_values)]
                env.set_velocity_command(vx, args.vy, args.yaw)

            action = model.act_inference(obs)
            obs, reward, done, _ = env.step(action)

            vx_error = torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])
            reward_sum += reward.mean().item()
            done_sum += done.float().mean().item()
            vx_error_sum += vx_error.mean().item()

            if step % args.print_interval == 0:
                elapsed_steps = step + 1
                print(
                    f"step={step:05d} "
                    f"cmd_vx={env.commands[:, 0].mean().item():.3f} "
                    f"measured_vx={env.base_lin_vel[:, 0].mean().item():.3f} "
                    f"vx_err={vx_error.mean().item():.3f} "
                    f"reward={reward.mean().item():.3f} "
                    f"done={done.float().mean().item():.3f} "
                    f"avg_reward={reward_sum / elapsed_steps:.3f} "
                    f"avg_vx_err={vx_error_sum / elapsed_steps:.3f} "
                    f"{format_action_stats(env)}"
                )
                if args.spread_vx:
                    print(format_per_env_action_stats(env))

    print("Finished rollout.")
    print(f"mean reward: {reward_sum / args.steps:.4f}")
    print(f"mean done: {done_sum / args.steps:.4f}")
    print(f"mean vx abs error: {vx_error_sum / args.steps:.4f}")
    time.sleep(0.5)


if __name__ == "__main__":
    main()
