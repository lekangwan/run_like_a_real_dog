import argparse
import csv
from pathlib import Path
import pickle as pkl
from types import SimpleNamespace
import time

import isaacgym

assert isaacgym
from isaacgym import gymtorch
import torch

from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from gait_conditions import DIRECTED_PUSH_INTERVAL_S, apply_condition_cfg
from gait_project_config import MAINLINE_TASK_MAP, TRAIN_MESH_TYPE
from train_high_level_oracle_ppo import (
    GAIT_NAMES,
    GAIT_SHORT_NAMES,
    read_task_specs,
)
from train_high_level_ppo import ActorCritic, find_logdir, load_low_level_policy


DEFAULT_ROUTE = (
    "flat_trot_efficiency:1.0,"
    "ramp_up_trot_robustness:1.0,"
    "rough_slope_trot_robustness:1.0,"
    "push_lateral_pace_recovery:1.5,"
    "stepping_stones_easy_bound_highspeed:2.0"
)


def latest_checkpoint(run_dir):
    checkpoints = sorted(Path(run_dir).glob("checkpoints/high_level_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No high_level_*.pt checkpoints found under: {run_dir}")
    return checkpoints[-1]


def latest_run_dir(save_root):
    candidates = []
    for checkpoint in Path(save_root).glob("*/checkpoints/high_level_*.pt"):
        candidates.append((checkpoint.stat().st_mtime, checkpoint.parent.parent))
    if not candidates:
        raise FileNotFoundError(f"No oracle checkpoints found under: {save_root}")
    return sorted(candidates)[-1][1]


def load_model(checkpoint_path, obs_dim, num_gaits, residual_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ActorCritic(obs_dim, num_gaits, residual_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("iteration", -1))


def parse_route(text, specs):
    by_task = {spec.task_id: spec for spec in specs}
    route = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            task_id, vx_text = item.split(":", 1)
            vx = float(vx_text)
        else:
            task_id = item
            vx = None
        if task_id not in by_task:
            raise ValueError(f"Unknown task_id={task_id!r}. Choices: {sorted(by_task)}")
        spec = by_task[task_id]
        if vx is None:
            vx = 0.5 * (spec.vx_low + spec.vx_high)
        route.append(
            SimpleNamespace(
                task_id=spec.task_id,
                condition=spec.condition,
                target_gait=spec.target_gait,
                target_gait_id=spec.target_gait_id,
                task_index=specs.index(spec),
                vx=vx,
                task_reward_weights=spec.task_reward_weights,
            )
        )
    if not route:
        raise ValueError("Route is empty")
    return route


def load_route_low_level_env(
    logdir,
    route,
    render,
    segment_length,
    terrain_width,
    mesh_type,
):
    config_go2(Cfg)
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        loaded_cfg = pkl_cfg["Cfg"]
        for key, value in loaded_cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in value.items():
                    setattr(getattr(Cfg, key), key2, value2)

    apply_condition_cfg(Cfg, "flat")
    Cfg.env.num_envs = 1
    Cfg.env.num_recording_envs = 0
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = len(route)
    Cfg.terrain.num_cols = 1
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.measure_heights = True
    Cfg.terrain.mesh_type = mesh_type
    Cfg.terrain.terrain_length = segment_length
    Cfg.terrain.terrain_width = terrain_width
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.edge_reset_robots = False
    Cfg.terrain.env_conditions = [item.condition for item in route]
    Cfg.asset.flip_visual_attachments = True

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
    Cfg.domain_rand.max_push_vel_xy = 1.5
    Cfg.domain_rand.push_axis = 1

    print("Creating route terrain:")
    for idx, item in enumerate(route):
        print(
            f"  segment={idx} condition={item.condition} task={item.task_id} "
            f"target={item.target_gait} vx={item.vx:.2f}"
        )
    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=not render, cfg=Cfg)
    return HistoryWrapper(env)


def base_env_from(wrapper):
    env = wrapper
    while hasattr(env, "env"):
        env = env.env
    return env


def set_robot_start(high_env, segment_length, start_x):
    base = high_env._get_base_env()
    origin_y = 0.5 * base.cfg.terrain.terrain_width
    origin_z = float(base.cfg.terrain.env_origins[0, 0, 2])
    base.root_states[0] = base.base_init_state.to(base.device)
    base.root_states[0, 0] = start_x
    base.root_states[0, 1] = origin_y
    base.root_states[0, 2] = base.base_init_state[2].to(base.device) + origin_z
    base.root_states[0, 7:13] = 0.0
    base.env_origins[0, 0] = 0.5 * segment_length
    base.env_origins[0, 1] = origin_y
    base.env_origins[0, 2] = origin_z
    base.gym.set_actor_root_state_tensor(base.sim, gymtorch.unwrap_tensor(base.root_states))
    base.gym.refresh_actor_root_state_tensor(base.sim)


def segment_index(x_pos, segment_length, num_segments):
    idx = int(x_pos // segment_length)
    return max(0, min(num_segments - 1, idx))


def update_segment_context(high_env, route, segment_id):
    item = route[segment_id]
    high_env.set_velocity_command(item.vx, 0.0, 0.0)
    high_env.set_target_gait(torch.tensor([item.target_gait_id], device=high_env.device))
    high_env.set_task_reward_weights(torch.tensor([item.task_reward_weights], device=high_env.device))

    base = high_env._get_base_env()
    is_push = item.condition == "push_lateral"
    base.cfg.domain_rand.push_robots = is_push
    base.cfg.domain_rand.push_axis = 1
    base.cfg.domain_rand.push_interval_s = DIRECTED_PUSH_INTERVAL_S
    base.cfg.domain_rand.push_interval = max(1, int(round(DIRECTED_PUSH_INTERVAL_S / base.dt)))
    base.cfg.domain_rand.max_push_vel_xy = 1.5


def augment_obs(obs, route, segment_id, device, num_tasks):
    task_index = route[segment_id].task_index
    one_hot = torch.zeros(1, num_tasks, device=device)
    one_hot[0, task_index] = 1.0
    return torch.cat((obs, one_hot), dim=-1)


def make_stats():
    return {
        "steps": 0,
        "reward_sum": 0.0,
        "done_sum": 0.0,
        "vx_err_sum": 0.0,
        "lateral_offset_sum": 0.0,
        "clip_sum": 0.0,
        "frequency_sum": 0.0,
        "duration_sum": 0.0,
        "footswing_sum": 0.0,
        "stance_width_sum": 0.0,
        "body_pitch_sum": 0.0,
        "gait_counts": {name: 0 for name in GAIT_NAMES},
        "last_gait_id": None,
        "gait_switch_count": 0,
    }


def add_step_stats(stats, high_env, action, reward, done):
    mapped = high_env._map_action(action)
    gait_id = int(torch.argmax(mapped["selector_weights"][0]).item())
    base = high_env._get_base_env()
    lateral_offset = torch.abs(base.root_states[0, 1] - base.env_origins[0, 1])
    stats["steps"] += 1
    stats["reward_sum"] += float(reward[0].item())
    stats["done_sum"] += float(done[0].float().item())
    stats["vx_err_sum"] += float(torch.abs(high_env.base_lin_vel[0, 0] - high_env.commands[0, 0]).item())
    stats["lateral_offset_sum"] += float(lateral_offset.item())
    stats["clip_sum"] += float((torch.abs(action[0, high_env.num_gaits :]) > 0.98).float().mean().item())
    stats["frequency_sum"] += float(mapped["frequency"][0].item())
    stats["duration_sum"] += float(mapped["duration"][0].item())
    stats["footswing_sum"] += float(mapped["footswing_height"][0].item())
    stats["stance_width_sum"] += float(mapped["stance_width"][0].item())
    stats["body_pitch_sum"] += float(mapped["body_pitch"][0].item())
    stats["gait_counts"][high_env.gait_names[gait_id]] += 1
    if stats["last_gait_id"] is not None and stats["last_gait_id"] != gait_id:
        stats["gait_switch_count"] += 1
    stats["last_gait_id"] = gait_id


def finalize_stats(route, all_stats):
    rows = []
    for idx, item in enumerate(route):
        stats = all_stats[idx]
        steps = max(1, stats["steps"])
        row = {
            "segment": idx,
            "task_id": item.task_id,
            "condition": item.condition,
            "target_gait": item.target_gait,
            "cmd_vx": item.vx,
            "steps": stats["steps"],
            "reward_mean": stats["reward_sum"] / steps,
            "done_rate": stats["done_sum"] / steps,
            "vx_err_mean": stats["vx_err_sum"] / steps,
            "lateral_offset_mean": stats["lateral_offset_sum"] / steps,
            "action_clip_rate": stats["clip_sum"] / steps,
            "frequency_mean": stats["frequency_sum"] / steps,
            "duration_mean": stats["duration_sum"] / steps,
            "footswing_height_mean": stats["footswing_sum"] / steps,
            "stance_width_mean": stats["stance_width_sum"] / steps,
            "body_pitch_mean": stats["body_pitch_sum"] / steps,
            "gait_switch_rate": stats["gait_switch_count"] / max(1, stats["steps"] - 1),
        }
        for gait_name in GAIT_NAMES:
            row[f"{GAIT_SHORT_NAMES[gait_name]}_ratio"] = stats["gait_counts"][gait_name] / steps
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_row(row):
    print(
        f"segment={row['segment']} condition={row['condition']} target={row['target_gait']} "
        f"steps={row['steps']} reward={row['reward_mean']:.3f} "
        f"vx_err={row['vx_err_mean']:.3f} lateral={row['lateral_offset_mean']:.3f} "
        f"done={row['done_rate']:.3f} "
        f"gaits[p/t/b/pa]={row['pronk_ratio']:.2f}/"
        f"{row['trot_ratio']:.2f}/{row['bound_ratio']:.2f}/{row['pace_ratio']:.2f} "
        f"switch={row['gait_switch_rate']:.3f} "
        f"freq={row['frequency_mean']:.2f} foot={row['footswing_height_mean']:.3f} "
        f"width={row['stance_width_mean']:.3f} pitch={row['body_pitch_mean']:.3f} "
        f"clip={row['action_clip_rate']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-root", default="runs/high_level_oracle_gait")
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--task-map", default=str(MAINLINE_TASK_MAP))
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--segment-length", type=float, default=8.0)
    parser.add_argument("--terrain-width", type=float, default=8.0)
    parser.add_argument("--start-x", type=float, default=1.5)
    parser.add_argument("--finish-margin", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--mesh-type", default=TRAIN_MESH_TYPE, choices=["heightfield", "trimesh"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    specs = read_task_specs(args.task_map, style_reward_scale=0.0)
    route = parse_route(args.route, specs)
    num_tasks = len(specs)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        run_dir = checkpoint_path.parent.parent
    else:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(args.save_root)
        checkpoint_path = latest_checkpoint(run_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logdir = find_logdir(args.label, args.run_index)
    low_policy = load_low_level_policy(logdir)
    low_env = load_route_low_level_env(
        logdir,
        route,
        render=not args.no_render,
        segment_length=args.segment_length,
        terrain_width=args.terrain_width,
        mesh_type=args.mesh_type,
    )
    high_env = HighLevelGaitWrapper(low_env, low_policy, record_reward_terms=True)
    device = high_env.device
    model, checkpoint_iteration = load_model(
        checkpoint_path,
        high_env.num_high_level_obs_history + num_tasks,
        high_env.num_gaits,
        high_env.num_behavior_actions,
        device,
    )

    obs = high_env.reset()
    set_robot_start(high_env, args.segment_length, args.start_x)
    current_segment = segment_index(args.start_x, args.segment_length, len(route))
    update_segment_context(high_env, route, current_segment)
    high_env.obs_history.zero_()
    obs = augment_obs(high_env.get_observations(), route, current_segment, device, num_tasks)

    output_dir = Path(args.output_dir) if args.output_dir else Path(run_dir) / "route_tests" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "route_timeseries.csv"
    summary_path = output_dir / "route_summary.csv"

    print(f"Loaded high-level checkpoint: {checkpoint_path} iteration={checkpoint_iteration}")
    print(f"Loaded low-level run: {logdir}")
    print(f"Writing route test outputs to: {output_dir}")
    print("Note: this is an oracle-conditioned policy test; the task one-hot switches at route segment boundaries.")

    all_stats = [make_stats() for _ in route]
    timeseries_rows = []
    total_length = len(route) * args.segment_length
    finish_x = total_length - args.finish_margin

    with torch.inference_mode():
        for step in range(args.max_steps):
            x_pos = float(high_env.root_states[0, 0].item())
            segment = segment_index(x_pos, args.segment_length, len(route))
            if segment != current_segment:
                current_segment = segment
                update_segment_context(high_env, route, current_segment)
                print(
                    f"Entered segment {current_segment}: condition={route[current_segment].condition} "
                    f"task={route[current_segment].task_id} vx={route[current_segment].vx:.2f}"
                )

            action = model.act_inference(obs)
            next_base_obs, reward, done, info = high_env.step(action)
            actual_action = info.get("executed_high_level_action", high_env.high_level_action).detach().clone()
            add_step_stats(all_stats[current_segment], high_env, actual_action, reward, done)

            mapped = high_env._map_action(actual_action)
            gait_id = int(torch.argmax(mapped["selector_weights"][0]).item())
            base = high_env._get_base_env()
            lateral_offset = torch.abs(base.root_states[0, 1] - base.env_origins[0, 1])
            timeseries_rows.append(
                {
                    "step": step,
                    "x": float(high_env.root_states[0, 0].item()),
                    "segment": current_segment,
                    "task_id": route[current_segment].task_id,
                    "condition": route[current_segment].condition,
                    "cmd_vx": float(high_env.commands[0, 0].item()),
                    "measured_vx": float(high_env.base_lin_vel[0, 0].item()),
                    "lateral_offset": float(lateral_offset.item()),
                    "reward": float(reward[0].item()),
                    "done": int(done[0].item()),
                    "gait": high_env.gait_names[gait_id],
                    "frequency": float(mapped["frequency"][0].item()),
                    "duration": float(mapped["duration"][0].item()),
                    "footswing_height": float(mapped["footswing_height"][0].item()),
                    "stance_width": float(mapped["stance_width"][0].item()),
                    "body_pitch": float(mapped["body_pitch"][0].item()),
                    "action_clip_rate": float(
                        (torch.abs(actual_action[0, high_env.num_gaits :]) > 0.98).float().mean().item()
                    ),
                }
            )

            if step % args.print_interval == 0:
                print(
                    f"step={step:05d} x={timeseries_rows[-1]['x']:.2f}/{finish_x:.2f} "
                    f"segment={current_segment} condition={route[current_segment].condition} "
                    f"gait={timeseries_rows[-1]['gait']} "
                    f"cmd_vx={timeseries_rows[-1]['cmd_vx']:.2f} "
                    f"measured_vx={timeseries_rows[-1]['measured_vx']:.2f} "
                    f"lat={timeseries_rows[-1]['lateral_offset']:.2f} "
                    f"reward={timeseries_rows[-1]['reward']:.3f}"
                )

            if bool(done[0].item()):
                print(f"Episode ended at step={step}, segment={current_segment}, x={timeseries_rows[-1]['x']:.2f}")
                break
            if timeseries_rows[-1]["x"] >= finish_x:
                print(f"Finished route at step={step}, x={timeseries_rows[-1]['x']:.2f}")
                break

            obs = augment_obs(next_base_obs, route, current_segment, device, num_tasks)

    rows = finalize_stats(route, all_stats)
    write_csv(timeseries_path, timeseries_rows)
    write_csv(summary_path, rows)
    print("\nRoute summary:")
    for row in rows:
        print_row(row)
    print(f"\nWrote: {timeseries_path}")
    print(f"Wrote: {summary_path}")
    time.sleep(0.5)


if __name__ == "__main__":
    main()
