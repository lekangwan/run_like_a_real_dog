import argparse
from pathlib import Path
import time

import isaacgym

assert isaacgym
import pandas as pd
import torch

from build_gait_template_library import metric_directions, objective_weights, score_candidates, viability_filter
from evaluate_gait_templates import build_command_tensor, load_env, set_commands
from scan_gait_params import find_logdir, load_policy


def parse_floats(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float")
    return values


def parse_strings(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one string")
    return values


def select_template(eval_results, condition, vx, gait, objective, min_forward_ratio, max_done_rate):
    candidates = eval_results[
        (eval_results["condition"] == condition)
        & ((eval_results["vx"].astype(float) - vx).abs() < 1e-6)
        & (eval_results["gait"] == gait)
    ].copy()
    if candidates.empty:
        raise ValueError(f"No candidates for condition={condition}, vx={vx}, gait={gait}")

    weights = objective_weights().get(objective, {})
    directions = metric_directions()
    filtered, was_filtered = viability_filter(candidates, min_forward_ratio, max_done_rate)
    scored = score_candidates(filtered, weights, directions)
    best = scored.sort_values(["template_objective_score", "template_score"], ascending=False).iloc[0]
    row = best.to_dict()
    row["viability_filtered"] = was_filtered
    return row


def build_batch(eval_results, args):
    batch = []
    selected = []
    env_id = 0
    vx_values = args.vx_list if args.vx_list is not None else [args.vx]
    template_condition = args.template_condition or args.condition
    for vx in vx_values:
        for _ in range(args.repeats):
            for gait in args.gaits:
                template = select_template(
                    eval_results,
                    template_condition,
                    vx,
                    gait,
                    args.objective,
                    args.min_forward_ratio,
                    args.max_done_rate,
                )
                params = {
                    "vx": vx,
                    "gait": gait,
                    "frequency": float(template["frequency"]),
                    "duration": float(template.get("duration", 0.5)),
                    "footswing_height": float(template["footswing_height"]),
                    "body_pitch": float(template["body_pitch"]),
                    "stance_width": float(template["stance_width"]),
                }
                batch.append(params)
                selected.append(
                    {
                        "env_id": env_id,
                        "condition": args.condition,
                        "template_condition": template_condition,
                        "objective": args.objective,
                        **params,
                        "template_objective_score": template.get("template_objective_score", ""),
                        "template_score": template.get("template_score", ""),
                        "measured_vx": template.get("measured_vx", ""),
                        "forward_distance_ratio": template.get("forward_distance_ratio", ""),
                        "progress_deficit": template.get("progress_deficit", ""),
                        "lateral_vel_rms": template.get("lateral_vel_rms", ""),
                        "roll_rate_rms": template.get("roll_rate_rms", ""),
                        "yaw_rate_rms": template.get("yaw_rate_rms", ""),
                        "orientation_penalty": template.get("orientation_penalty", ""),
                        "viability_filtered": template.get("viability_filtered", ""),
                    }
                )
                env_id += 1
    return batch, selected


def print_selected(selected):
    print("Selected templates for simultaneous gait comparison:")
    for row in selected:
        print(
            f"  env={int(row['env_id']):02d} "
            f"vx={float(row['vx']):.2f} gait={row['gait']} "
            f"freq={float(row['frequency']):.2f} "
            f"duration={float(row['duration']):.2f} "
            f"footswing={float(row['footswing_height']):.3f} "
            f"body_pitch={float(row['body_pitch']):.3f} "
            f"stance_width={float(row['stance_width']):.3f} "
            f"obj_score={float(row['template_objective_score']):.3f} "
            f"fwd_ratio={float(row['forward_distance_ratio']):.3f}"
        )


def group_indices(selected, device):
    groups = {}
    for row in selected:
        groups.setdefault(row["gait"], []).append(int(row["env_id"]))
    return {gait: torch.tensor(ids, device=device, dtype=torch.long) for gait, ids in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--eval-results", default="logs/gait_condition_eval_v6_selected/template_eval_results.csv")
    parser.add_argument("--condition", default="push_hard")
    parser.add_argument(
        "--template-condition",
        default=None,
        help="Condition used to select templates from eval-results. Defaults to --condition.",
    )
    parser.add_argument("--objective", default="push_bound_recovery")
    parser.add_argument("--gaits", type=parse_strings, default=parse_strings("bounding,trotting"))
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vx-list", type=parse_floats, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--min-forward-ratio", type=float, default=0.45)
    parser.add_argument("--max-done-rate", type=float, default=0.02)
    parser.add_argument("--output-dir", default="logs/push_hard_gait_compare")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_results)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {eval_path}")
    eval_results = pd.read_csv(eval_path)

    batch, selected = build_batch(eval_results, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(output_dir / "selected_templates.csv", index=False)

    logdir = find_logdir(args.label, args.run_index)
    env = load_env(logdir, len(batch), headless=args.no_render, condition=args.condition)
    policy = load_policy(logdir)
    obs = env.reset()

    commands = build_command_tensor(env, batch, len(batch))
    set_commands(env, commands)
    groups = group_indices(selected, env.commands.device)

    print(f"Loaded run: {logdir}")
    print(
        f"condition={args.condition}, template_condition={args.template_condition or args.condition}, "
        f"objective={args.objective}, "
        f"num_envs={len(batch)}, render={not args.no_render}"
    )
    print_selected(selected)
    print(f"Saved selected template table to: {output_dir / 'selected_templates.csv'}")
    if args.condition == "push_hard":
        print("Note: push_hard samples random pushes per env, so paired envs share condition strength but not identical impulses.")
    elif args.condition.startswith("push_"):
        print("Note: directed push conditions apply the same fixed push velocity to paired envs.")

    reward_sum = {gait: 0.0 for gait in groups}
    vx_err_sum = {gait: 0.0 for gait in groups}
    lateral_sum = {gait: 0.0 for gait in groups}
    done_sum = {gait: 0.0 for gait in groups}

    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            set_commands(env, commands)
            obs, reward, dones, _ = env.step(actions)
            vx_error = torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])
            lateral_vel = torch.abs(env.base_lin_vel[:, 1])

            for gait, ids in groups.items():
                reward_sum[gait] += reward[ids].mean().item()
                vx_err_sum[gait] += vx_error[ids].mean().item()
                lateral_sum[gait] += lateral_vel[ids].mean().item()
                done_sum[gait] += dones[ids].float().mean().item()

            if step % args.print_interval == 0:
                elapsed = step + 1
                parts = []
                for gait, ids in groups.items():
                    parts.append(
                        f"{gait}: vx={env.base_lin_vel[ids, 0].mean().item():.3f} "
                        f"vx_err={vx_error[ids].mean().item():.3f} "
                        f"lat={lateral_vel[ids].mean().item():.3f} "
                        f"done={dones[ids].float().mean().item():.3f} "
                        f"avg_reward={reward_sum[gait] / elapsed:.3f}"
                    )
                print(f"step={step:05d} | " + " | ".join(parts))

    print("Finished gait comparison.")
    for gait in groups:
        print(
            f"{gait}: mean_reward={reward_sum[gait] / args.steps:.4f}, "
            f"mean_vx_err={vx_err_sum[gait] / args.steps:.4f}, "
            f"mean_lateral_vel={lateral_sum[gait] / args.steps:.4f}, "
            f"mean_done={done_sum[gait] / args.steps:.4f}"
        )
    time.sleep(0.5)


if __name__ == "__main__":
    main()
