import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import isaacgym

assert isaacgym
import numpy as np
import torch

from gait_project_config import (
    MAINLINE_TASK_MAP,
    TRAIN_EDGE_RESET_MARGIN,
    TRAIN_MESH_TYPE,
    TRAIN_TELEPORT_THRESH,
    TRAIN_TERRAIN_SIZE,
)
from train_high_level_oracle_ppo import (
    GAIT_NAMES,
    OracleConditionHighLevelEnv,
    load_selector_target_table,
    parse_residual_action_mask,
    read_task_specs,
)
from train_high_level_ppo import ActorCritic, find_logdir, load_low_level_policy
from evaluate_high_level_policy_by_task import latest_checkpoint, load_run_args


TRAINING_RANGE_EVAL = (
    "flat_trot_efficiency:0.5,"
    "flat_trot_efficiency:1.0,"
    "flat_trot_efficiency:1.5,"
    "flat_trot_efficiency:2.0,"
    "ramp_up_trot_robustness:0.5,"
    "ramp_up_trot_robustness:1.0,"
    "ramp_up_trot_robustness:1.5,"
    "ramp_up_trot_robustness:2.0,"
    "rough_slope_trot_robustness:0.5,"
    "rough_slope_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.5,"
    "rough_slope_trot_robustness:2.0,"
    "push_lateral_pace_recovery:1.2,"
    "push_lateral_pace_recovery:1.5,"
    "push_lateral_pace_recovery:1.8,"
    "stepping_stones_easy_bound_highspeed:1.7,"
    "stepping_stones_easy_bound_highspeed:2.0"
)


def parse_eval_items(text, specs):
    by_task = {spec.task_id: spec for spec in specs}
    items = []
    for raw_item in text.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        task_id, vx_text = raw_item.split(":", 1)
        if task_id not in by_task:
            raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
        spec = by_task[task_id]
        items.append((spec, specs.index(spec), float(vx_text)))
    if not items:
        raise ValueError("No eval items requested")
    return items


def set_fixed_vx(env, vx):
    env.vx_cmd[:] = vx
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def load_model(checkpoint_path, env, run_args):
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    obs_dim = int(run_args.get("obs_dim", env.obs_dim))
    model = ActorCritic(
        obs_dim,
        env.num_gaits,
        env.num_behavior_actions,
        base_obs_dim=env.base_obs_dim,
        priv_dim=int(run_args.get("priv_dim", 14)),
        z_dim=int(run_args.get("z_dim", 16)),
        selector_latent_cmd_only=bool(run_args.get("selector_latent_cmd_only", False)),
        physical_aux_dim=int(run_args.get("physical_aux_dim", 0)),
        selector_physical_state_input=bool(run_args.get("selector_physical_state_input", False)),
        adaptation_temporal_summary=bool(run_args.get("adaptation_temporal_summary", False)),
        gait_conditioned_residuals=bool(run_args.get("gait_conditioned_residuals", False)),
        gait_input_residuals=bool(run_args.get("gait_input_residuals", False)),
    ).to(env.device)
    model.load_state_dict(checkpoint["model"])
    residual_mask = run_args.get("residual_action_mask")
    if residual_mask is None:
        residual_mask = parse_residual_action_mask(run_args.get("residual_train_dims", "all"), device=env.device)
    else:
        residual_mask = torch.tensor(residual_mask, device=env.device)
    model.set_residual_action_mask(residual_mask)
    model.eval()
    return model, int(checkpoint.get("iteration", -1))


def augment_for_checkpoint(obs, task_index, num_tasks, use_task_id):
    if not use_task_id:
        return obs
    one_hot = torch.zeros(obs.shape[0], num_tasks, device=obs.device, dtype=obs.dtype)
    one_hot[:, task_index] = 1.0
    return torch.cat((obs, one_hot), dim=-1)


def append_command_vx_obs(obs, cmd_vx, enabled):
    if not enabled:
        return obs
    return torch.cat((obs, cmd_vx[:, None].to(dtype=obs.dtype)), dim=-1)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def append_batch(storage, **items):
    for key, value in items.items():
        storage.setdefault(key, []).append(value)


def concat_storage(storage):
    output = {}
    for key, values in storage.items():
        first = values[0]
        if isinstance(first, np.ndarray):
            output[key] = np.concatenate(values, axis=0)
        else:
            output[key] = np.array([item for value in values for item in value])
    return output


def gait_probs_from_z(model, obs, z):
    policy_obs = torch.cat((obs, z), dim=-1)
    logits, _ = model.distribution_params(policy_obs)
    return torch.softmax(logits, dim=-1)


def collect_item(args, spec, task_index, all_specs, logdir, low_policy, checkpoint_path, run_args):
    use_task_id = not bool(run_args.get("no_oracle_condition_obs", False))
    selector_only = bool(run_args.get("selector_only", False))
    selector_latent_cmd_only = bool(run_args.get("selector_latent_cmd_only", False))

    env = OracleConditionHighLevelEnv(
        [spec],
        logdir,
        low_policy,
        args.num_envs,
        render=args.render,
        oracle_condition_obs=False,
        terrain_size=args.terrain_size,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
        selector_hold_steps=int(run_args.get("selector_hold_steps", 3)),
    )
    model, checkpoint_iteration = load_model(checkpoint_path, env, run_args)
    selector_table = load_selector_target_table(
        args.selector_targets,
        all_specs,
        env.device,
        env.num_gaits,
        min_confidence=args.selector_aux_min_confidence,
    )

    obs = augment_for_checkpoint(env.reset(), task_index, len(all_specs), use_task_id)
    set_fixed_vx(env, args.current_vx)
    obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)
    storage = {}
    collected = 0

    with torch.inference_mode():
        for step in range(args.max_steps):
            base_obs = env.get_base_obs()
            priv_obs = env.get_high_level_privileged_obs()

            if model.z_dim > 0:
                z_student = model.encode_student(base_obs)
                z_teacher = model.encode_teacher(priv_obs)
                probs_student = gait_probs_from_z(model, obs, z_student)
                probs_teacher = gait_probs_from_z(model, obs, z_teacher)
                probs_zero = gait_probs_from_z(model, obs, torch.zeros_like(z_student))
                probs_shuffle = gait_probs_from_z(model, obs, torch.roll(z_student, shifts=1, dims=0))
            else:
                z_student = torch.zeros(base_obs.shape[0], 0, device=env.device)
                z_teacher = torch.zeros(base_obs.shape[0], 0, device=env.device)
                logits, _ = model.actor.distribution_params(obs)
                probs_student = torch.softmax(logits, dim=-1)
                probs_teacher = probs_student
                probs_zero = probs_student
                probs_shuffle = probs_student

            if selector_table is not None:
                target_probs, target_weights = selector_table.lookup(
                    torch.full((env.num_envs,), task_index, device=env.device, dtype=torch.long),
                    env.command_vx(),
                )
            else:
                target_probs = torch.zeros(env.num_envs, env.num_gaits, device=env.device)
                target_weights = torch.zeros(env.num_envs, device=env.device)

            action = model.act_student_selector_only(obs) if selector_only else model.act_student(obs)
            next_obs, _reward, done, info = env.step(action)
            set_fixed_vx(env, args.current_vx)

            if step >= args.warmup_steps and (step - args.warmup_steps) % args.sample_interval == 0:
                executed = info.get("executed_high_level_action", action)
                task_ids = np.full(env.num_envs, task_index, dtype=np.int64)
                cmd_vx = np.full(env.num_envs, args.current_vx, dtype=np.float32)
                append_batch(
                    storage,
                    history=tensor_to_numpy(base_obs),
                    privileged_obs=tensor_to_numpy(priv_obs),
                    z_student=tensor_to_numpy(z_student),
                    z_teacher=tensor_to_numpy(z_teacher),
                    gait_probs_student=tensor_to_numpy(probs_student),
                    gait_probs_teacher=tensor_to_numpy(probs_teacher),
                    gait_probs_zero_z=tensor_to_numpy(probs_zero),
                    gait_probs_shuffled_z=tensor_to_numpy(probs_shuffle),
                    selector_target_probs=tensor_to_numpy(target_probs),
                    selector_target_weight=tensor_to_numpy(target_weights).reshape(-1),
                    task_index=task_ids,
                    cmd_vx=cmd_vx,
                    selected_gait=tensor_to_numpy(torch.argmax(action[:, : env.num_gaits], dim=-1)).astype(np.int64),
                    executed_gait=tensor_to_numpy(torch.argmax(executed[:, : env.num_gaits], dim=-1)).astype(np.int64),
                    done=tensor_to_numpy(done.float()).reshape(-1),
                )
                collected += env.num_envs
                if collected >= args.samples_per_item:
                    break

            obs = augment_for_checkpoint(next_obs, task_index, len(all_specs), use_task_id)
            obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)

    data = concat_storage(storage)
    data["checkpoint_iteration"] = np.full(data["task_index"].shape[0], checkpoint_iteration, dtype=np.int64)
    del env, model, obs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return data


def write_item_summary(path, summaries):
    if not summaries:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def child_dir_name(task_id, vx):
    return f"{task_id}_vx{vx:.2f}".replace(".", "p")


def combine_child_outputs(output_dir, child_dirs, metadata_overrides):
    all_data = {}
    summaries = []
    first_metadata = None
    for child_dir in child_dirs:
        child_dir = Path(child_dir)
        data = np.load(child_dir / "info_path_samples.npz", allow_pickle=True)
        for key in data.files:
            all_data.setdefault(key, []).append(data[key])
        with (child_dir / "collection_summary.csv").open(newline="") as file:
            summaries.extend(csv.DictReader(file))
        if first_metadata is None:
            first_metadata = json.loads((child_dir / "metadata.json").read_text())

    arrays = {key: np.concatenate(values, axis=0) for key, values in all_data.items()}
    np.savez_compressed(output_dir / "info_path_samples.npz", **arrays)
    write_item_summary(output_dir / "collection_summary.csv", summaries)

    metadata = dict(first_metadata or {})
    metadata.update(metadata_overrides)
    metadata["num_samples"] = int(arrays["task_index"].shape[0])
    metadata["child_dirs"] = [str(Path(path)) for path in child_dirs]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def run_child_collections(args, eval_items, output_dir):
    child_dirs = []
    for spec, _task_index, vx in eval_items:
        child_dir = output_dir / child_dir_name(spec.task_id, vx)
        child_dirs.append(child_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-dir",
            args.run_dir,
            "--label",
            args.label,
            "--run-index",
            str(args.run_index),
            "--task-map",
            args.task_map,
            "--eval",
            f"{spec.task_id}:{vx}",
            "--num-envs",
            str(args.num_envs),
            "--samples-per-item",
            str(args.samples_per_item),
            "--warmup-steps",
            str(args.warmup_steps),
            "--max-steps",
            str(args.max_steps),
            "--sample-interval",
            str(args.sample_interval),
            "--terrain-size",
            str(args.terrain_size),
            "--edge-reset-margin",
            str(args.edge_reset_margin),
            "--teleport-thresh",
            str(args.teleport_thresh),
            "--mesh-type",
            args.mesh_type,
            "--output-dir",
            str(child_dir),
            "--no-spawn",
        ]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.selector_targets:
            cmd += ["--selector-targets", args.selector_targets]
        if args.selector_aux_min_confidence:
            cmd += ["--selector-aux-min-confidence", str(args.selector_aux_min_confidence)]
        if args.render:
            cmd.append("--render")
        print(f"[spawn] task={spec.task_id} vx={vx:.2f}")
        subprocess.run(cmd, check=True)

    combine_child_outputs(
        output_dir,
        child_dirs,
        {
            "eval": args.eval,
            "num_child_runs": len(child_dirs),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--eval", default=TRAINING_RANGE_EVAL)
    parser.add_argument("--selector-targets", default=None)
    parser.add_argument("--selector-aux-min-confidence", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--samples-per-item", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--sample-interval", type=int, default=4)
    parser.add_argument("--terrain-size", type=float, default=TRAIN_TERRAIN_SIZE)
    parser.add_argument("--edge-reset-margin", type=float, default=TRAIN_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=TRAIN_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-spawn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run_dir)
    run_args = load_run_args(run_dir)
    specs = read_task_specs(args.task_map, style_reward_scale=0.0)
    eval_items = parse_eval_items(args.eval, specs)
    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "info_path_probe" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(eval_items) > 1 and not args.no_spawn:
        run_child_collections(args, eval_items, output_dir)
        print(f"Wrote combined: {output_dir / 'info_path_samples.npz'}")
        print(f"Wrote combined: {output_dir / 'metadata.json'}")
        print(f"Wrote combined: {output_dir / 'collection_summary.csv'}")
        return

    all_data = {}
    summaries = []
    task_id_by_index = {index: spec.task_id for index, spec in enumerate(specs)}
    condition_by_index = {index: spec.condition for index, spec in enumerate(specs)}

    for spec, task_index, vx in eval_items:
        print(f"[collect] task={spec.task_id} vx={vx:.2f}")
        args.current_vx = vx
        item_data = collect_item(args, spec, task_index, specs, logdir, low_policy, checkpoint_path, run_args)
        for key, value in item_data.items():
            all_data.setdefault(key, []).append(value)
        summaries.append(
            {
                "task_id": spec.task_id,
                "condition": spec.condition,
                "cmd_vx": vx,
                "samples": int(item_data["task_index"].shape[0]),
            }
        )

    arrays = {key: np.concatenate(values, axis=0) for key, values in all_data.items()}
    target_top = np.argmax(arrays["selector_target_probs"], axis=1).astype(np.int64)
    target_top[arrays["selector_target_weight"] <= 0.0] = -1
    arrays["selector_target_top_gait"] = target_top

    np.savez_compressed(output_dir / "info_path_samples.npz", **arrays)
    write_item_summary(output_dir / "collection_summary.csv", summaries)

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "run_args": run_args,
        "task_id_by_index": task_id_by_index,
        "condition_by_index": condition_by_index,
        "gait_names": GAIT_NAMES,
        "selector_targets": args.selector_targets,
        "selector_aux_min_confidence": args.selector_aux_min_confidence,
        "num_samples": int(arrays["task_index"].shape[0]),
        "eval": args.eval,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Wrote: {output_dir / 'info_path_samples.npz'}")
    print(f"Wrote: {output_dir / 'metadata.json'}")
    print(f"Wrote: {output_dir / 'collection_summary.csv'}")


if __name__ == "__main__":
    main()
