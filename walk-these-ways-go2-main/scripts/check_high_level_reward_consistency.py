import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import isaacgym

assert isaacgym
import torch

from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper
from go2_gym.envs.wrappers.high_level_reward_metrics import (
    compute_metric_score_dict,
    compute_weighted_metric_reward,
    stack_metric_scores,
)
from gait_project_config import MAINLINE_TASK_MAP
from train_high_level_oracle_ppo import (
    GAIT_NAMES,
    OracleConditionHighLevelEnv,
    find_logdir,
    load_low_level_policy,
    read_task_specs,
)


DEFAULT_EVAL = "flat_trot_efficiency:1.0,stepping_stones_easy_bound_highspeed:2.0"
DEFAULT_RESIDUAL_SETS = "zero=0,0,0,0,0;high_clearance=0,0,1,1,0"


def parse_eval_items(eval_text):
    items = []
    for token in str(eval_text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Expected task:vx token, got {token!r}")
        task_id, vx_text = token.split(":", 1)
        items.append((task_id.strip(), float(vx_text)))
    return items


def parse_residual_sets(text):
    sets = []
    for token in str(text).split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            name, values_text = token.split("=", 1)
            name = name.strip()
        else:
            name = f"set{len(sets)}"
            values_text = token
        values = [float(item.strip()) for item in values_text.split(",") if item.strip()]
        if len(values) != 5:
            raise ValueError(f"Residual set {name!r} must contain 5 values, got {values}")
        sets.append((name, torch.tensor(values, dtype=torch.float)))
    if not sets:
        raise ValueError("At least one residual set is required.")
    return sets


def residual_set_arg(name, values):
    return f"{name}=" + ",".join(f"{float(value):.8g}" for value in values.tolist())


def safe_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace(":", "_")
        .replace(",", "_")
        .replace("=", "_")
        .replace(".", "p")
    )


def clone_spec_for_fixed_vx(spec, vx):
    cloned = argparse.Namespace(**vars(spec))
    cloned.vx_values = [float(vx)]
    cloned.vx_low = float(vx)
    cloned.vx_high = float(vx)
    return cloned


def build_action(env, gait_name, residual_values):
    gait_id = GAIT_NAMES.index(gait_name)
    action = torch.zeros(
        env.num_envs,
        env.num_high_level_actions,
        device=env.device,
        dtype=torch.float,
    )
    action[:, gait_id] = 1.0
    action[:, env.num_gaits :] = residual_values.to(device=env.device, dtype=torch.float)
    return action


def update_error(stats, metric, error_tensor):
    error = error_tensor.detach().abs().flatten()
    if error.numel() == 0:
        return
    item = stats.setdefault(metric, {"max": 0.0, "sum": 0.0, "count": 0})
    item["max"] = max(item["max"], float(torch.max(error).item()))
    item["sum"] += float(torch.sum(error).item())
    item["count"] += int(error.numel())


def offline_from_primitives(primitives, task_reward_weights):
    score_sums = None
    reward_sum = None
    primitive_sums = {}

    for primitive in primitives:
        score_dict = compute_metric_score_dict(**primitive)
        scores = stack_metric_scores(score_dict, HighLevelGaitWrapper.TASK_REWARD_NAMES)
        weighted_reward = compute_weighted_metric_reward(scores, task_reward_weights)

        if score_sums is None:
            score_sums = {
                name: torch.zeros_like(value)
                for name, value in score_dict.items()
            }
            reward_sum = torch.zeros_like(weighted_reward)
            primitive_sums = {
                key: torch.zeros_like(value)
                for key, value in primitive.items()
            }

        for name, value in score_dict.items():
            score_sums[name] += value
        for key, value in primitive.items():
            primitive_sums[key] += value
        reward_sum += weighted_reward

    denom = max(len(primitives), 1)
    score_avg = {name: value / denom for name, value in score_sums.items()}
    primitive_avg = {name: value / denom for name, value in primitive_sums.items()}
    reward_avg = reward_sum / denom
    return score_avg, primitive_avg, reward_avg


def run_case(args, spec, vx, gait_name, residual_name, residual_values, low_policy, logdir):
    env = OracleConditionHighLevelEnv(
        [clone_spec_for_fixed_vx(spec, vx)],
        logdir,
        low_policy,
        num_envs=args.num_envs,
        render=False,
        oracle_condition_obs=True,
        selector_hold_steps=0,
        terrain_size=args.terrain_size,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
    )
    env.env.record_reward_primitives = True
    env.env.record_reward_terms = True
    env.reset()
    action = build_action(env, gait_name, residual_values)

    for _ in range(args.warmup_steps):
        env.step(action)

    error_stats = {}
    for _ in range(args.steps):
        _, reward, _, info = env.step(action)
        online_terms = info["high_level_reward_terms"]
        primitives = info["high_level_reward_primitives"]
        offline_scores, offline_primitives, offline_reward = offline_from_primitives(
            primitives,
            env.env.task_reward_weights,
        )

        update_error(
            error_stats,
            "weighted_metric_reward",
            offline_reward - online_terms["weighted_metric_reward"],
        )
        update_error(error_stats, "returned_reward", offline_reward - reward)
        for name in HighLevelGaitWrapper.TASK_REWARD_NAMES:
            update_error(
                error_stats,
                f"score_{name}",
                offline_scores[name] - online_terms[f"score_{name}"],
            )
        for name, value in offline_primitives.items():
            if name in online_terms:
                update_error(error_stats, name, value - online_terms[name])

    rows = []
    for metric, item in sorted(error_stats.items()):
        count = max(item["count"], 1)
        rows.append(
            {
                "task_id": spec.task_id,
                "cmd_vx": vx,
                "gait": gait_name,
                "residual_set": residual_name,
                "metric": metric,
                "max_abs_error": item["max"],
                "mean_abs_error": item["sum"] / count,
                "count": item["count"],
            }
        )
    return rows


def write_outputs(output_dir, rows, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reward_consistency.csv"
    fieldnames = [
        "task_id",
        "cmd_vx",
        "gait",
        "residual_set",
        "metric",
        "max_abs_error",
        "mean_abs_error",
        "count",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    max_error = max((row["max_abs_error"] for row in rows), default=0.0)
    passed = max_error <= args.tolerance
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as file:
        file.write("# High-Level Reward Consistency Check\n\n")
        file.write(f"- reward_profile: `{args.reward_profile}`\n")
        file.write(f"- tolerance: `{args.tolerance}`\n")
        file.write(f"- max_abs_error: `{max_error:.8g}`\n")
        file.write(f"- passed: `{passed}`\n\n")
        file.write("This check compares online `HighLevelGaitWrapper` reward terms\n")
        file.write("against offline recomputation from the same recorded trajectory\n")
        file.write("primitives using `high_level_reward_metrics.py`.\n\n")
        file.write("## Files\n\n")
        file.write("- `reward_consistency.csv`\n")

    with open(output_dir / "args.json", "w") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)

    return passed, max_error, csv_path, summary_path


def run_child_cases(args, eval_items, gait_names, residual_sets):
    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    rows = []

    for task_id, vx in eval_items:
        for gait_name in gait_names:
            for residual_name, residual_values in residual_sets:
                case_name = (
                    f"{safe_name(task_id)}_vx{safe_name(f'{vx:.2f}')}_"
                    f"{safe_name(gait_name)}_{safe_name(residual_name)}"
                )
                case_dir = cases_dir / case_name
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--label",
                    args.label,
                    "--run-index",
                    str(args.run_index),
                    "--task-map",
                    args.task_map,
                    "--eval",
                    f"{task_id}:{vx}",
                    "--gaits",
                    gait_name,
                    "--residual-sets",
                    residual_set_arg(residual_name, residual_values),
                    "--reward-profile",
                    args.reward_profile,
                    "--num-envs",
                    str(args.num_envs),
                    "--steps",
                    str(args.steps),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--terrain-size",
                    str(args.terrain_size),
                    "--edge-reset-margin",
                    str(args.edge_reset_margin),
                    "--teleport-thresh",
                    str(args.teleport_thresh),
                    "--mesh-type",
                    args.mesh_type,
                    "--tolerance",
                    str(args.tolerance),
                    "--output-dir",
                    str(case_dir),
                    "--no-spawn",
                ]
                print(
                    f"[spawn] task={task_id} vx={vx:.2f} gait={gait_name} "
                    f"residual_set={residual_name}"
                )
                subprocess.run(cmd, check=True)
                csv_path = case_dir / "reward_consistency.csv"
                with open(csv_path, newline="") as file:
                    rows.extend(csv.DictReader(file))

    for row in rows:
        row["max_abs_error"] = float(row["max_abs_error"])
        row["mean_abs_error"] = float(row["mean_abs_error"])
        row["count"] = int(row["count"])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Check online/offline consistency for high-level reward metrics."
    )
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--gaits", default="pronking,trotting,bounding,pacing")
    parser.add_argument("--residual-sets", default=DEFAULT_RESIDUAL_SETS)
    parser.add_argument("--reward-profile", default="canonical_efficiency_candidate")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--terrain-size", type=float, default=12.0)
    parser.add_argument("--edge-reset-margin", type=float, default=1.5)
    parser.add_argument("--teleport-thresh", type=float, default=1.5)
    parser.add_argument("--mesh-type", default="trimesh")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output-dir",
        default="runs/high_level_oracle_gait/reward_consistency/latest",
    )
    parser.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    eval_items = parse_eval_items(args.eval)
    gait_names = [item.strip() for item in args.gaits.split(",") if item.strip()]
    residual_sets = parse_residual_sets(args.residual_sets)

    if not args.no_spawn:
        rows = run_child_cases(args, eval_items, gait_names, residual_sets)
        passed, max_error, csv_path, summary_path = write_outputs(Path(args.output_dir), rows, args)
        print(f"[done] passed={passed} max_abs_error={max_error:.8g}")
        print(f"[done] wrote {csv_path}")
        print(f"[done] wrote {summary_path}")
        if not passed:
            raise SystemExit(1)
        return

    all_specs = read_task_specs(args.task_map, reward_profile=args.reward_profile)
    specs_by_id = {spec.task_id: spec for spec in all_specs}
    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    rows = []
    for task_id, vx in eval_items:
        if task_id not in specs_by_id:
            raise ValueError(f"Unknown task_id={task_id!r}; available={sorted(specs_by_id)}")
        spec = specs_by_id[task_id]
        for gait_name in gait_names:
            if gait_name not in GAIT_NAMES:
                raise ValueError(f"Unknown gait={gait_name!r}; available={GAIT_NAMES}")
            for residual_name, residual_values in residual_sets:
                print(
                    f"[case] task={task_id} vx={vx:.2f} gait={gait_name} "
                    f"residual_set={residual_name}"
                )
                rows.extend(
                    run_case(
                        args,
                        spec,
                        vx,
                        gait_name,
                        residual_name,
                        residual_values,
                        low_policy,
                        logdir,
                    )
                )

    passed, max_error, csv_path, summary_path = write_outputs(Path(args.output_dir), rows, args)
    print(f"[done] passed={passed} max_abs_error={max_error:.8g}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {summary_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
