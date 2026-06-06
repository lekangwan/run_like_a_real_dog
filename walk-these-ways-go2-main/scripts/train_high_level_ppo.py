import argparse
import copy
import csv
import gc
import glob
import json
from pathlib import Path
import pickle as pkl
import time

import isaacgym

assert isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym.envs.wrappers.high_level_gait_wrapper import HighLevelGaitWrapper


def find_logdir(label, run_index):
    dirs = sorted(glob.glob(str(Path(MINI_GYM_ROOT_DIR) / "runs" / label / "*")))
    if not dirs:
        raise FileNotFoundError(f"No runs found for label: {label}")
    return dirs[run_index]


def load_low_level_policy(logdir):
    body = torch.jit.load(str(Path(logdir) / "checkpoints" / "body_latest.jit"))
    adaptation_module = torch.jit.load(str(Path(logdir) / "checkpoints" / "adaptation_module_latest.jit"))
    body.eval()
    adaptation_module.eval()

    def policy(obs, info=None):
        del info
        with torch.inference_mode():
            obs_history = obs["obs_history"].detach().cpu()
            latent = adaptation_module.forward(obs_history)
            action = body.forward(torch.cat((obs_history, latent), dim=-1))
        return action

    return policy


def load_low_level_env(logdir, num_envs, render):
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


class HybridActor(nn.Module):
    def __init__(self, obs_dim, num_gaits, residual_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(last_dim, hidden_dim), nn.ELU()]
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.gait_head = nn.Linear(last_dim, num_gaits)
        self.residual_head = nn.Linear(last_dim, residual_dim)
        self.num_gaits = num_gaits

    def distribution_params(self, obs):
        features = self.backbone(obs)
        gait_logits = self.gait_head(features)
        residual_mean = torch.tanh(self.residual_head(features))
        return gait_logits, residual_mean

    def forward(self, obs):
        gait_logits, residual_mean = self.distribution_params(obs)
        gait_ids = torch.argmax(gait_logits, dim=-1)
        gait_one_hot = F.one_hot(gait_ids, num_classes=self.num_gaits).to(dtype=residual_mean.dtype)
        return torch.cat((gait_one_hot, residual_mean), dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_gaits, residual_dim, hidden_dims=(256, 256), init_std=0.5):
        super().__init__()
        self.actor = HybridActor(obs_dim, num_gaits, residual_dim, hidden_dims)
        self.num_gaits = num_gaits
        self.residual_dim = residual_dim

        critic_layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            critic_layers += [nn.Linear(last_dim, hidden_dim), nn.ELU()]
            last_dim = hidden_dim
        critic_layers.append(nn.Linear(last_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.log_std = nn.Parameter(torch.log(torch.ones(residual_dim) * init_std))

    def distribution(self, obs):
        gait_logits, residual_mean = self.actor.distribution_params(obs)
        residual_std = torch.exp(self.log_std).expand_as(residual_mean)
        return Categorical(logits=gait_logits), Normal(residual_mean, residual_std)

    def act(self, obs):
        gait_dist, residual_dist = self.distribution(obs)
        gait_id = gait_dist.sample()
        gait_one_hot = F.one_hot(gait_id, num_classes=self.num_gaits).to(dtype=obs.dtype)
        residual_action = torch.clamp(residual_dist.sample(), -1.0, 1.0)
        action = torch.cat((gait_one_hot, residual_action), dim=-1)
        log_prob = gait_dist.log_prob(gait_id) + residual_dist.log_prob(residual_action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        gait_dist, residual_dist = self.distribution(obs)
        gait_id = torch.argmax(actions[:, : self.num_gaits], dim=-1)
        residual_action = actions[:, self.num_gaits :]
        log_prob = gait_dist.log_prob(gait_id) + residual_dist.log_prob(residual_action).sum(dim=-1)
        entropy = gait_dist.entropy() + residual_dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def act_inference(self, obs):
        return self.actor(obs)


class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.step].copy_(obs.detach())
        self.actions[self.step].copy_(action.detach())
        self.log_probs[self.step].copy_(log_prob.detach())
        self.rewards[self.step].copy_(reward.detach())
        self.dones[self.step].copy_(done.float().detach())
        self.values[self.step].copy_(value.detach())
        self.step += 1

    def compute_returns(self, last_value, gamma, lam):
        last_value = last_value.detach()
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if step == self.rewards.shape[0] - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        self.returns = self.returns.detach()
        self.advantages = self.advantages.detach()

    def flat(self):
        return (
            self.obs.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.log_probs.flatten(0, 1),
            self.returns.flatten(0, 1),
            self.advantages.flatten(0, 1),
            self.values.flatten(0, 1),
        )


def sample_vx(num_envs, device, low=0.2, high=1.0):
    return torch.rand(num_envs, device=device) * (high - low) + low


def save_checkpoint(path, model, optimizer, iteration):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"]) + 1


def append_metrics(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def add_high_level_action_metrics(metrics, env, vx_low, vx_high, num_bins=4, compact_metrics=False):
    with torch.inference_mode():
        mapped = env._map_action(env.high_level_action)
        target_selector_weights, target_residual_actions = env._speed_conditioned_target()
        target_mapped = env._map_target(target_selector_weights, target_residual_actions)
        gait_metric_names = {
            "pronking": "pronk",
            "trotting": "trot",
            "bounding": "bound",
            "pacing": "pace",
        }
        action_names = [
            "phase",
            "offset",
            "bound",
            "frequency",
            "duration",
            "footswing_height",
            "stance_width",
            "body_pitch",
        ]
        selector_weights = mapped["selector_weights"]
        target_selector_weights = target_mapped["selector_weights"]

        if compact_metrics:
            for gait_id, gait_name in enumerate(env.gait_names):
                short_name = gait_metric_names[gait_name]
                metrics[f"sel_{short_name}"] = selector_weights[:, gait_id].mean().item()
                metrics[f"target_sel_{short_name}"] = (
                    target_selector_weights[:, gait_id].mean().item()
                )

            metrics["cmd_phase"] = mapped["phase"].mean().item()
            metrics["cmd_offset"] = mapped["offset"].mean().item()
            metrics["cmd_bound"] = mapped["bound"].mean().item()
            metrics["cmd_freq"] = mapped["frequency"].mean().item()
            metrics["cmd_duration"] = mapped["duration"].mean().item()
            metrics["cmd_swing"] = mapped["footswing_height"].mean().item()
            metrics["cmd_stance"] = mapped["stance_width"].mean().item()
            metrics["cmd_pitch"] = mapped["body_pitch"].mean().item()

            vx_cmd = env.commands[:, 0]
            bin_width = (vx_high - vx_low) / num_bins
            for bin_id in range(num_bins):
                low = vx_low + bin_id * bin_width
                high = vx_low + (bin_id + 1) * bin_width
                if bin_id == num_bins - 1:
                    mask = (vx_cmd >= low) & (vx_cmd <= high)
                else:
                    mask = (vx_cmd >= low) & (vx_cmd < high)

                count = int(mask.sum().item())
                prefix = f"b{bin_id}"
                metrics[f"{prefix}_count"] = count
                for gait_id, gait_name in enumerate(env.gait_names):
                    short_name = gait_metric_names[gait_name]
                    metrics[f"{prefix}_sel_{short_name}"] = (
                        selector_weights[:, gait_id][mask].mean().item()
                        if count > 0
                        else float("nan")
                    )
                metrics[f"{prefix}_target_trot"] = (
                    target_selector_weights[:, 1][mask].mean().item() if count > 0 else float("nan")
                )
                metrics[f"{prefix}_target_bound"] = (
                    target_selector_weights[:, 2][mask].mean().item() if count > 0 else float("nan")
                )
            return

        for gait_id, gait_name in enumerate(env.gait_names):
            metrics[f"selector/{gait_name}_mean"] = selector_weights[:, gait_id].mean().item()
            metrics[f"target_selector/{gait_name}_mean"] = (
                target_selector_weights[:, gait_id].mean().item()
            )
        for name in action_names:
            metrics[f"action/{name}_mean"] = mapped[name].mean().item()
            metrics[f"target/{name}_mean"] = target_mapped[name].mean().item()

        vx_cmd = env.commands[:, 0]
        bin_width = (vx_high - vx_low) / num_bins
        for bin_id in range(num_bins):
            low = vx_low + bin_id * bin_width
            high = vx_low + (bin_id + 1) * bin_width
            if bin_id == num_bins - 1:
                mask = (vx_cmd >= low) & (vx_cmd <= high)
            else:
                mask = (vx_cmd >= low) & (vx_cmd < high)

            count = int(mask.sum().item())
            prefix = f"vxbin{bin_id}_{low:.2f}_{high:.2f}"
            metrics[f"{prefix}/count"] = count
            for gait_id, gait_name in enumerate(env.gait_names):
                metrics[f"{prefix}/selector_{gait_name}"] = (
                    selector_weights[:, gait_id][mask].mean().item() if count > 0 else float("nan")
                )
                metrics[f"{prefix}/target_selector_{gait_name}"] = (
                    target_selector_weights[:, gait_id][mask].mean().item()
                    if count > 0
                    else float("nan")
                )
            for name in action_names:
                metrics[f"{prefix}/{name}"] = (
                    mapped[name][mask].mean().item() if count > 0 else float("nan")
                )
                metrics[f"{prefix}/target_{name}"] = (
                    target_mapped[name][mask].mean().item() if count > 0 else float("nan")
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-go2/train")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--vx-low", type=float, default=0.2)
    parser.add_argument("--vx-high", type=float, default=1.0)
    parser.add_argument("--save-dir", default=str(Path(MINI_GYM_ROOT_DIR) / "runs" / "high_level_gait"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--compact-metrics", action="store_true")
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    logdir = find_logdir(args.label, args.run_index)
    low_env = load_low_level_env(logdir, args.num_envs, args.render)
    low_policy = load_low_level_policy(logdir)
    env = HighLevelGaitWrapper(low_env, low_policy)

    device = env.device
    model = ActorCritic(
        env.num_high_level_obs_history,
        env.num_gaits,
        env.num_behavior_actions,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    obs = env.reset()
    vx_cmd = sample_vx(args.num_envs, device, args.vx_low, args.vx_high)
    env.set_velocity_command(vx_cmd, 0.0, 0.0)

    if args.resume:
        resume_path = Path(args.resume)
        run_dir = resume_path.parents[1]
        run_name = run_dir.name
    else:
        resume_path = None
        run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.save_dir) / run_name
    metrics_path = run_dir / "metrics.csv"
    print(f"Saving high-level checkpoints to: {run_dir}")
    print(f"obs_dim={env.num_high_level_obs_history}, action_dim={env.num_high_level_actions}")
    run_dir.mkdir(parents=True, exist_ok=True)
    start_iteration = 0
    if resume_path:
        start_iteration = load_checkpoint(resume_path, model, optimizer, device)
        print(f"Resumed from {resume_path} at iteration {start_iteration}")

    args_dict = vars(args).copy()
    args_dict["resolved_run_name"] = run_name
    args_dict["start_iteration"] = start_iteration
    with open(run_dir / "args.json", "w") as file:
        json.dump(args_dict, file, indent=2)

    end_iteration = start_iteration + args.iterations
    for iteration in range(start_iteration, end_iteration):
        if args.log_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        buffer = RolloutBuffer(
            args.num_steps,
            args.num_envs,
            env.num_high_level_obs_history,
            env.num_high_level_actions,
            device,
        )

        reward_sum = 0.0
        done_sum = 0.0
        vx_error_sum = 0.0
        vx_mean_sum = 0.0

        for _ in range(args.num_steps):
            with torch.inference_mode():
                action, log_prob, value = model.act(obs)
                next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, log_prob, reward, done, value)

            done_ids = done.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                vx_cmd[done_ids] = sample_vx(len(done_ids), device, args.vx_low, args.vx_high)
                env.set_velocity_command(vx_cmd, 0.0, 0.0)

            reward_sum += reward.mean().item()
            done_sum += done.float().mean().item()
            vx_error_sum += torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0]).mean().item()
            vx_mean_sum += env.base_lin_vel[:, 0].mean().item()
            obs = next_obs

        with torch.inference_mode():
            last_value = model.critic(obs).squeeze(-1)
        buffer.compute_returns(last_value, args.gamma, args.lam)

        flat_obs, flat_actions, flat_log_probs, flat_returns, flat_advantages, flat_values = buffer.flat()
        batch_size = flat_obs.shape[0]
        mini_batch_size = batch_size // args.mini_batches

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        updates = 0

        for _ in range(args.epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start : start + mini_batch_size]
                new_log_prob, entropy, value = model.evaluate_actions(flat_obs[idx], flat_actions[idx])
                ratio = torch.exp(new_log_prob - flat_log_probs[idx])
                surrogate_1 = ratio * flat_advantages[idx]
                surrogate_2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * flat_advantages[idx]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = (flat_returns[idx] - value).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy_loss.item()
                updates += 1

        metrics = {
            "iteration": iteration,
            "reward": reward_sum / args.num_steps,
            "done": done_sum / args.num_steps,
            "vx": vx_mean_sum / args.num_steps,
            "vx_err": vx_error_sum / args.num_steps,
            "policy_loss": policy_loss_epoch / updates,
            "value_loss": value_loss_epoch / updates,
            "entropy": entropy_epoch / updates,
            "log_std_mean": model.log_std.detach().mean().item(),
        }
        add_high_level_action_metrics(
            metrics,
            env,
            args.vx_low,
            args.vx_high,
            compact_metrics=args.compact_metrics,
        )

        if iteration % args.save_interval == 0 or iteration == end_iteration - 1:
            save_checkpoint(run_dir / "checkpoints" / f"high_level_{iteration:06d}.pt", model, optimizer, iteration)
            actor_cpu = copy.deepcopy(model.actor).cpu()
            scripted_actor = torch.jit.script(actor_cpu)
            scripted_actor.save(str(run_dir / "checkpoints" / "high_level_actor_latest.jit"))
            del actor_cpu, scripted_actor

        del (
            buffer,
            flat_obs,
            flat_actions,
            flat_log_probs,
            flat_returns,
            flat_advantages,
            flat_values,
            indices,
            idx,
            new_log_prob,
            entropy,
            value,
            ratio,
            surrogate_1,
            surrogate_2,
            policy_loss,
            value_loss,
            entropy_loss,
            loss,
            last_value,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        if args.log_memory and torch.cuda.is_available():
            metrics["cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            metrics["cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
            metrics["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

        append_metrics(metrics_path, metrics)

        memory_text = ""
        if "cuda_allocated_mb" in metrics:
            memory_text = (
                f" cuda_alloc={metrics['cuda_allocated_mb']:.0f}MB"
                f" cuda_reserved={metrics['cuda_reserved_mb']:.0f}MB"
                f" cuda_peak={metrics['cuda_max_allocated_mb']:.0f}MB"
            )

        print(
            f"iter={iteration:04d} "
            f"reward={metrics['reward']:.3f} "
            f"done={metrics['done']:.3f} "
            f"vx={metrics['vx']:.3f} "
            f"vx_err={metrics['vx_err']:.3f} "
            f"policy_loss={metrics['policy_loss']:.3f} "
            f"value_loss={metrics['value_loss']:.3f} "
            f"entropy={metrics['entropy']:.3f}"
            f"{memory_text}"
        )


if __name__ == "__main__":
    main()
