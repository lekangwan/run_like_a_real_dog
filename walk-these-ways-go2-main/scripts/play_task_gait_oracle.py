import argparse
from pathlib import Path
import time

import isaacgym

assert isaacgym
import pandas as pd
import torch

from evaluate_gait_templates import build_command_tensor, load_env, set_commands
from scan_gait_params import find_logdir, load_policy


def parse_floats(value):
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float")
    return values


def select_template(library, task_id, condition, vx):
    subset = library.copy()
    if task_id:
        subset = subset[subset["task_id"] == task_id]
    if condition:
        subset = subset[subset["condition"] == condition]
    if subset.empty:
        raise ValueError(
            f"No template found for task_id={task_id!r}, condition={condition!r}. "
            "Check gait_template_library.csv."
        )
    distances = (subset["vx"].astype(float) - vx).abs()
    return subset.loc[distances.idxmin()].to_dict()


def build_batch(library, args):
    vx_values = args.vx_list if args.vx_list is not None else [args.vx]
    if args.spread_vx:
        repeated = []
        while len(repeated) < args.num_envs:
            repeated.extend(vx_values)
        env_vx = repeated[: args.num_envs]
    else:
        env_vx = [vx_values[0]] * args.num_envs

    batch = []
    selected = []
    for vx in env_vx:
        template = select_template(library, args.task_id, args.condition, vx)
        params = {
            "vx": vx,
            "gait": template["target_gait"],
            "frequency": float(template["frequency"]),
            "duration": float(template.get("duration", 0.5)),
            "footswing_height": float(template["footswing_height"]),
            "body_pitch": float(template["body_pitch"]),
            "stance_width": float(template["stance_width"]),
        }
        batch.append(params)
        selected.append(template)
    return batch, selected


def infer_condition(library, task_id, condition):
    if condition:
        return condition
    subset = library[library["task_id"] == task_id]
    if subset.empty:
        raise ValueError(f"Cannot infer condition for unknown task_id={task_id!r}")
    conditions = sorted(subset["condition"].unique())
    if len(conditions) != 1:
        raise ValueError(
            f"task_id={task_id!r} maps to multiple conditions {conditions}; pass --condition."
        )
    return conditions[0]


def print_selection(selected, max_rows=8):
    print("Selected oracle templates:")
    for i, template in enumerate(selected[:max_rows]):
        print(
            f"  env={i:02d} task={template['task_id']} "
            f"condition={template['condition']} gait={template['target_gait']} "
            f"source_vx={float(template['source_vx']):.2f} "
            f"freq={float(template['frequency']):.2f} "
            f"duration={float(template.get('duration', 0.5)):.2f} "
            f"footswing={float(template['footswing_height']):.3f} "
            f"body_pitch={float(template['body_pitch']):.3f} "
            f"stance_width={float(template['stance_width']):.3f} "
            f"score={float(template['template_objective_score']):.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--library",
        default="logs/gait_condition_eval_v8_mainline/gait_template_library/gait_template_library.csv",
    )
    parser.add_argument("--task-id", default="flat_trot_efficiency")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.8)
    parser.add_argument("--vx-list", type=parse_floats, default=None)
    parser.add_argument("--spread-vx", action="store_true")
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--terrain-length", type=float, default=None)
    parser.add_argument("--terrain-width", type=float, default=None)
    parser.add_argument("--teleport-thresh", type=float, default=None)
    parser.add_argument("--edge-reset-margin", type=float, default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    library_path = Path(args.library)
    if not library_path.exists():
        raise FileNotFoundError(
            f"Template library not found: {library_path}. "
            "Run scripts/build_gait_template_library.py first."
        )
    library = pd.read_csv(library_path)
    args.condition = infer_condition(library, args.task_id, args.condition)

    batch, selected = build_batch(library, args)
    logdir = find_logdir(args.label, args.run_index)
    env = load_env(
        logdir,
        args.num_envs,
        headless=args.no_render,
        condition=args.condition,
        terrain_length=args.terrain_length,
        terrain_width=args.terrain_width,
        teleport_thresh=args.teleport_thresh,
        edge_reset_margin=args.edge_reset_margin,
    )
    policy = load_policy(logdir)

    obs = env.reset()
    commands = build_command_tensor(env, batch, args.num_envs)
    set_commands(env, commands)

    print(f"Loaded run: {logdir}")
    print(
        f"Oracle task={args.task_id}, condition={args.condition}, "
        f"num_envs={args.num_envs}, render={not args.no_render}"
    )
    print_selection(selected)

    reward_sum = 0.0
    done_sum = 0.0
    vx_error_sum = 0.0
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            set_commands(env, commands)
            obs, reward, dones, _ = env.step(actions)
            vx_error = torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])

            reward_sum += reward.mean().item()
            done_sum += dones.float().mean().item()
            vx_error_sum += vx_error.mean().item()

            if step % args.print_interval == 0:
                elapsed = step + 1
                print(
                    f"step={step:05d} "
                    f"cmd_vx={env.commands[:, 0].mean().item():.3f} "
                    f"measured_vx={env.base_lin_vel[:, 0].mean().item():.3f} "
                    f"vx_err={vx_error.mean().item():.3f} "
                    f"reward={reward.mean().item():.3f} "
                    f"done={dones.float().mean().item():.3f} "
                    f"avg_reward={reward_sum / elapsed:.3f} "
                    f"avg_vx_err={vx_error_sum / elapsed:.3f}"
                )

    print("Finished oracle rollout.")
    print(f"mean reward: {reward_sum / args.steps:.4f}")
    print(f"mean done: {done_sum / args.steps:.4f}")
    print(f"mean vx abs error: {vx_error_sum / args.steps:.4f}")
    time.sleep(0.5)


if __name__ == "__main__":
    main()
