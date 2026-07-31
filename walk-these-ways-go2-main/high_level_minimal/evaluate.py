import argparse
import csv
import json
from pathlib import Path

import isaacgym

assert isaacgym
import torch

from .config import LOW_LEVEL_LABEL, TASK_MAP, validate_decision_interval
from .environment import HighLevelEnvironment
from .low_level import find_run
from .model import HighLevelPolicy
from .tasks import fixed_gait_action, parse_eval_specs


def parse_args():
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
    candidates = sorted((run_dir / "checkpoints").glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {run_dir / 'checkpoints'}")
    return candidates[-1]


def load_run_config(run_dir):
    minimal_path = run_dir / "args_minimal.json"
    if minimal_path.exists():
        with open(minimal_path) as file:
            return json.load(file), True
    with open(run_dir / "args.json") as file:
        return json.load(file), False


def create_model(env, config, minimal_run):
    if minimal_run:
        stage = config["stage"]
        gait_input = stage == "parameters"
        selector_only = stage == "gait"
    else:
        gait_input = bool(config.get("gait_input_residuals", False))
        selector_only = bool(config.get("selector_only", False))

    model = HighLevelPolicy(
        policy_obs_dim=env.policy_obs_dim,
        base_obs_dim=env.base_obs_dim,
        num_gaits=env.num_gaits,
        residual_dim=env.action_dim - env.num_gaits,
        privileged_dim=int(config.get("priv_dim", 14)),
        latent_dim=int(config.get("z_dim", config.get("latent_dim", 16))),
        use_gait_input_residuals=gait_input,
    ).to(env.device)
    model.residual_action_mask.fill_(0.0 if selector_only else 1.0)
    return model, selector_only


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config, minimal_run = load_run_config(run_dir)
    saved_interval = int(config.get("decision_interval", 1))
    decision_interval = args.decision_interval or saved_interval
    validate_decision_interval(decision_interval)

    specs = parse_eval_specs(args.eval, TASK_MAP)
    low_level_run = Path(
        config["low_level_run"]
        if "low_level_run" in config
        else find_run(LOW_LEVEL_LABEL)
    )
    env = HighLevelEnvironment(
        specs,
        low_level_run,
        args.num_envs,
        render=args.render,
    )
    observation = env.reset()
    model, selector_only = create_model(env, config, minimal_run)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = (
        "actor.gait_input_residual_network.",
    )
    disallowed = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if disallowed or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={disallowed}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    task_ids = env.assignment.task_ids
    reward_sum = torch.zeros(len(specs), device=env.device)
    vx_error_sum = torch.zeros(len(specs), device=env.device)
    done_sum = torch.zeros(len(specs), device=env.device)
    gait_counts = torch.zeros(len(specs), env.num_gaits, device=env.device)
    sample_counts = torch.zeros(len(specs), device=env.device)

    for step in range(args.steps):
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
