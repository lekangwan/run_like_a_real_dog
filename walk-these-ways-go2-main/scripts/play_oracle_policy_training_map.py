import argparse
import json
from pathlib import Path
import time

import isaacgym

assert isaacgym
import torch

from gait_project_config import (
    MAINLINE_TASK_MAP,
    TRAIN_MESH_TYPE,
    VIS_EDGE_RESET_MARGIN,
    VIS_TERRAIN_LENGTH,
    VIS_TELEPORT_THRESH,
)
from train_high_level_oracle_ppo import (
    GAIT_SHORT_NAMES,
    OracleConditionHighLevelEnv,
    parse_residual_action_mask,
    residual_mask_description,
    read_task_specs,
)
from train_high_level_ppo import ActorCritic, find_logdir, load_low_level_policy


def latest_checkpoint(run_dir):
    checkpoints = sorted((Path(run_dir) / "checkpoints").glob("high_level_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No high_level_*.pt checkpoints found under: {run_dir}")
    return checkpoints[-1]


def load_run_args(run_dir):
    args_path = Path(run_dir) / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open() as file:
        return json.load(file)


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


def augment_for_checkpoint(obs, task_ids, num_tasks, use_task_id):
    if not use_task_id:
        return obs
    one_hot = torch.zeros(obs.shape[0], num_tasks, device=obs.device, dtype=obs.dtype)
    one_hot[torch.arange(obs.shape[0], device=obs.device), task_ids.to(device=obs.device, dtype=torch.long)] = 1.0
    return torch.cat((obs, one_hot), dim=-1)


def append_command_vx_obs(obs, cmd_vx, enabled):
    if not enabled:
        return obs
    return torch.cat((obs, cmd_vx[:, None].to(dtype=obs.dtype)), dim=-1)


def set_deterministic_vx(env):
    vx_cmd = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    for task_index, spec in enumerate(env.specs):
        env_ids = (env.assignment.task_ids == task_index).nonzero(as_tuple=False).flatten()
        values = getattr(spec, "vx_values", None) or [0.5 * (spec.vx_low + spec.vx_high)]
        values = torch.tensor(values, device=env.device, dtype=torch.float)
        repeats = (len(env_ids) + len(values) - 1) // len(values)
        vx_cmd[env_ids] = values.repeat(repeats)[: len(env_ids)]
    env.vx_cmd[:] = vx_cmd
    env.env.set_velocity_command(env.vx_cmd, 0.0, 0.0)


def print_scene_layout(env):
    print("Mixed training map layout:")
    for task_index, spec in enumerate(env.specs):
        env_ids = (env.assignment.task_ids == task_index).nonzero(as_tuple=False).flatten()
        vx_values = env.vx_cmd[env_ids].detach().cpu().numpy()
        print(
            f"  task={task_index} envs={env_ids.detach().cpu().tolist()} "
            f"condition={spec.condition} target={spec.target_gait} "
            f"vx={vx_values}"
        )


def print_policy_stats(env, max_envs):
    mapped = env.env._map_action(env.env.high_level_action)
    selector = mapped["selector_weights"]
    count = min(env.num_envs, max_envs)
    for env_id in range(count):
        task_index = int(env.assignment.task_ids[env_id].item())
        spec = env.specs[task_index]
        gait_id = int(torch.argmax(selector[env_id]).item())
        print(
            f"  env={env_id:02d} task={spec.task_id} "
            f"cmd_vx={env.command_vx()[env_id].item():.2f} "
            f"measured_vx={env.measured_vx()[env_id].item():.2f} "
            f"gait={GAIT_SHORT_NAMES[env.env.gait_names[gait_id]]} "
            f"freq={mapped['frequency'][env_id].item():.2f} "
            f"duration={mapped['duration'][env_id].item():.2f} "
            f"footswing={mapped['footswing_height'][env_id].item():.3f} "
            f"stance_width={mapped['stance_width'][env_id].item():.3f} "
            f"body_pitch={mapped['body_pitch'][env_id].item():.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--num-envs-per-task", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--print-interval", type=int, default=200)
    parser.add_argument("--max-print-envs", type=int, default=20)
    parser.add_argument("--terrain-size", type=float, default=VIS_TERRAIN_LENGTH)
    parser.add_argument("--edge-reset-margin", type=float, default=VIS_EDGE_RESET_MARGIN)
    parser.add_argument("--teleport-thresh", type=float, default=VIS_TELEPORT_THRESH)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--sample-vx", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None and args.run_dir is None:
        raise ValueError("Use either --checkpoint or --run-dir.")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parent.parent
    run_args = load_run_args(run_dir)
    oracle_condition_obs = not bool(run_args.get("no_oracle_condition_obs", False))
    selector_only = bool(run_args.get("selector_only", False))
    selector_latent_cmd_only = bool(run_args.get("selector_latent_cmd_only", False))

    specs = read_task_specs(args.task_map, style_reward_scale=0.0)
    num_envs = len(specs) * args.num_envs_per_task
    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    env = OracleConditionHighLevelEnv(
        specs,
        logdir,
        low_policy,
        num_envs,
        render=not args.no_render,
        oracle_condition_obs=False,
        terrain_size=args.terrain_size,
        edge_reset_margin=args.edge_reset_margin,
        teleport_thresh=args.teleport_thresh,
        mesh_type=args.mesh_type,
        selector_hold_steps=int(run_args.get("selector_hold_steps", 3)),
    )
    model, iteration = load_model(checkpoint_path, env, run_args)

    base_obs = env.reset()
    if not args.sample_vx:
        set_deterministic_vx(env)
    obs = augment_for_checkpoint(base_obs, env.assignment.task_ids, len(specs), oracle_condition_obs)
    obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)

    print(f"Loaded checkpoint: {checkpoint_path} iteration={iteration}")
    print(
        f"num_envs={env.num_envs}, obs_dim={env.obs_dim}, action_dim={env.num_high_level_actions}, "
        f"oracle_condition_obs={oracle_condition_obs}, selector_only={selector_only}, "
        f"selector_latent_cmd_only={selector_latent_cmd_only}"
    )
    print(
        "residual_train_dims="
        f"{residual_mask_description(model.residual_action_mask.detach().cpu())}"
    )
    print_scene_layout(env)

    reward_sum = 0.0
    done_sum = 0.0
    vx_error_sum = 0.0
    with torch.inference_mode():
        for step in range(args.steps):
            if selector_only:
                action = model.act_student_selector_only(obs)
            else:
                action = model.act_student(obs)
            next_base_obs, reward, done, _ = env.step(action)
            if not args.sample_vx:
                set_deterministic_vx(env)
            obs = augment_for_checkpoint(next_base_obs, env.assignment.task_ids, len(specs), oracle_condition_obs)
            obs = append_command_vx_obs(obs, env.command_vx(), selector_latent_cmd_only)
            vx_error = torch.abs(env.measured_vx() - env.command_vx())
            reward_sum += reward.mean().item()
            done_sum += done.float().mean().item()
            vx_error_sum += vx_error.mean().item()

            if step % args.print_interval == 0:
                elapsed = step + 1
                print(
                    f"step={step:05d} reward={reward.mean().item():.3f} "
                    f"avg_reward={reward_sum / elapsed:.3f} "
                    f"done={done.float().mean().item():.3f} "
                    f"avg_done={done_sum / elapsed:.3f} "
                    f"vx_err={vx_error.mean().item():.3f} "
                    f"avg_vx_err={vx_error_sum / elapsed:.3f}"
                )
                print_policy_stats(env, args.max_print_envs)

    print("Finished mixed training-map rollout.")
    print(f"mean reward: {reward_sum / args.steps:.4f}")
    print(f"mean done: {done_sum / args.steps:.4f}")
    print(f"mean vx abs error: {vx_error_sum / args.steps:.4f}")
    time.sleep(0.5)


if __name__ == "__main__":
    main()
