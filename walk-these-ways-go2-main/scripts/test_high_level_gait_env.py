import argparse
import glob
from pathlib import Path
import pickle as pkl

import isaacgym

assert isaacgym
import torch

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper


def parse_action(value):
    action = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(action) != 8:
        raise argparse.ArgumentTypeError("--action must contain 8 comma-separated values")
    return action


def find_logdir(label, run_index):
    dirs = sorted(glob.glob(str(Path(MINI_GYM_ROOT_DIR) / "runs" / label / "*")))
    if not dirs:
        raise FileNotFoundError(f"No runs found for label: {label}")
    return dirs[run_index]


def load_policy(logdir):
    body = torch.jit.load(str(Path(logdir) / "checkpoints" / "body_latest.jit"))
    adaptation_module = torch.jit.load(str(Path(logdir) / "checkpoints" / "adaptation_module_latest.jit"))

    def policy(obs, info=None):
        if info is None:
            info = {}
        obs_history = obs["obs_history"].to("cpu")
        latent = adaptation_module.forward(obs_history)
        action = body.forward(torch.cat((obs_history, latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(logdir, num_envs, render):
    config_go2(Cfg)
    with open(Path(logdir) / "parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in value.items():
                    setattr(getattr(Cfg, key), key2, value2)

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_envs = num_envs
    Cfg.env.num_recording_envs = 0
    Cfg.terrain.curriculum = False
    Cfg.terrain.selected = False
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 0
    Cfg.terrain.num_rows = 1
    Cfg.terrain.num_cols = max(1, num_envs)
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = False
    Cfg.terrain.teleport_robots = True
    Cfg.asset.flip_visual_attachments = True

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=not render, cfg=Cfg)
    return HistoryWrapper(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--action", type=parse_action, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    logdir = find_logdir(args.label, args.run_index)
    low_env = load_env(logdir, args.num_envs, args.render)
    low_policy = load_policy(logdir)
    env = HighLevelGaitWrapper(low_env, low_policy, record_reward_terms=True)

    obs = env.reset()
    env.set_velocity_command(args.vx, 0.0, 0.0)

    if args.action is None:
        high_action = torch.zeros(args.num_envs, env.num_high_level_actions, device=env.device)
    else:
        high_action = torch.tensor(args.action, device=env.device, dtype=torch.float).repeat(args.num_envs, 1)

    reward_sum = torch.zeros(args.num_envs, device=env.device)
    done_sum = torch.zeros(args.num_envs, device=env.device)
    vx_sum = torch.zeros(args.num_envs, device=env.device)
    vx_abs_error_sum = torch.zeros(args.num_envs, device=env.device)
    term_sums = {}
    for _ in range(args.steps):
        obs, reward, done, info = env.step(high_action)
        reward_sum += reward
        done_sum += done.float()
        vx_sum += env.base_lin_vel[:, 0]
        vx_abs_error_sum += torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])
        for key, value in info["high_level_reward_terms"].items():
            if key not in term_sums:
                term_sums[key] = torch.zeros(args.num_envs, device=env.device)
            term_sums[key] += value

    print(f"high-level obs shape: {tuple(obs.shape)}")
    print(f"mean reward: {reward_sum.mean().item() / args.steps:.4f}")
    print(f"mean done count: {done_sum.mean().item():.4f}")
    print(f"rollout mean measured vx: {vx_sum.mean().item() / args.steps:.4f}")
    print(f"final mean measured vx: {env.base_lin_vel[:, 0].mean().item():.4f}")
    print(f"mean vx abs error: {vx_abs_error_sum.mean().item() / args.steps:.4f}")
    for key in sorted(term_sums):
        print(f"mean {key}: {term_sums[key].mean().item() / args.steps:.4f}")


if __name__ == "__main__":
    main()
