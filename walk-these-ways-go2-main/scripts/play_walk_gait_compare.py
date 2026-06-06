import argparse
from pathlib import Path
import time

import isaacgym

assert isaacgym
import pandas as pd
import torch

from evaluate_gait_templates import build_command_tensor, load_env, set_commands
from scan_gait_params import GAIT_PARAMS, find_logdir, load_policy


def parse_floats(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float")
    return values


def parse_strings(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [value for value in values if value not in GAIT_PARAMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown gait(s): {unknown}. Choices: {sorted(GAIT_PARAMS)}")
    return values


def build_batch(args):
    vx_values = args.vx_list if args.vx_list is not None else [args.vx]
    batch = []
    selected = []
    env_id = 0
    for vx in vx_values:
        for gait in args.gaits:
            phase, offset, bound = GAIT_PARAMS[gait]
            params = {
                "vx": vx,
                "gait": gait,
                "frequency": args.frequency,
                "footswing_height": args.footswing_height,
                "body_pitch": args.body_pitch,
                "stance_width": args.stance_width,
                "duration": args.duration,
            }
            batch.append(params)
            selected.append(
                {
                    "env_id": env_id,
                    "condition": args.condition,
                    **params,
                    "phase": phase,
                    "offset": offset,
                    "bound": bound,
                }
            )
            env_id += 1
    return batch, selected


def group_indices(selected, device):
    groups = {}
    for row in selected:
        key = f"{row['gait']}@{float(row['vx']):.2f}"
        groups.setdefault(key, []).append(int(row["env_id"]))
    return {key: torch.tensor(ids, device=device, dtype=torch.long) for key, ids in groups.items()}


def print_selected(selected):
    print("Selected fixed gait commands:")
    for row in selected:
        print(
            f"  env={int(row['env_id']):02d} "
            f"condition={row['condition']} "
            f"vx={float(row['vx']):.2f} "
            f"gait={row['gait']} "
            f"freq={float(row['frequency']):.2f} "
            f"duration={float(row['duration']):.2f} "
            f"phase={float(row['phase']):.2f} "
            f"offset={float(row['offset']):.2f} "
            f"bound={float(row['bound']):.2f} "
            f"footswing={float(row['footswing_height']):.3f} "
            f"stance_width={float(row['stance_width']):.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--condition", default="rough_mid")
    parser.add_argument("--gaits", type=parse_strings, default=parse_strings("walking,trotting"))
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vx-list", type=parse_floats, default=None)
    parser.add_argument("--frequency", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.60)
    parser.add_argument("--footswing-height", type=float, default=0.08)
    parser.add_argument("--body-pitch", type=float, default=0.0)
    parser.add_argument("--stance-width", type=float, default=0.33)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--output-dir", default="logs/walk_gait_compare")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    batch, selected = build_batch(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(output_dir / "selected_commands.csv", index=False)

    logdir = find_logdir(args.label, args.run_index)
    env = load_env(logdir, len(batch), headless=args.no_render, condition=args.condition)
    policy = load_policy(logdir)
    obs = env.reset()

    commands = build_command_tensor(env, batch, len(batch))
    set_commands(env, commands)
    groups = group_indices(selected, env.commands.device)

    print(f"Loaded run: {logdir}")
    print(
        f"condition={args.condition}, num_envs={len(batch)}, render={not args.no_render}, "
        f"saved={output_dir / 'selected_commands.csv'}"
    )
    print_selected(selected)
    print("Walking is a candidate phase pattern; visual validation is required before using it as a training label.")

    reward_sum = {key: 0.0 for key in groups}
    vx_err_sum = {key: 0.0 for key in groups}
    lateral_sum = {key: 0.0 for key in groups}
    done_sum = {key: 0.0 for key in groups}

    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            set_commands(env, commands)
            obs, reward, dones, _ = env.step(actions)
            vx_error = torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])
            lateral_vel = torch.abs(env.base_lin_vel[:, 1])

            for key, ids in groups.items():
                reward_sum[key] += reward[ids].mean().item()
                vx_err_sum[key] += vx_error[ids].mean().item()
                lateral_sum[key] += lateral_vel[ids].mean().item()
                done_sum[key] += dones[ids].float().mean().item()

            if step % args.print_interval == 0:
                elapsed = step + 1
                parts = []
                for key, ids in groups.items():
                    parts.append(
                        f"{key}: vx={env.base_lin_vel[ids, 0].mean().item():.3f} "
                        f"vx_err={vx_error[ids].mean().item():.3f} "
                        f"lat={lateral_vel[ids].mean().item():.3f} "
                        f"done={dones[ids].float().mean().item():.3f} "
                        f"avg_reward={reward_sum[key] / elapsed:.3f}"
                    )
                print(f"step={step:05d} | " + " | ".join(parts))

    print("Finished walk gait comparison.")
    for key in groups:
        print(
            f"{key}: mean_reward={reward_sum[key] / args.steps:.4f}, "
            f"mean_vx_err={vx_err_sum[key] / args.steps:.4f}, "
            f"mean_lateral_vel={lateral_sum[key] / args.steps:.4f}, "
            f"mean_done={done_sum[key] / args.steps:.4f}"
        )
    time.sleep(0.5)


if __name__ == "__main__":
    main()
