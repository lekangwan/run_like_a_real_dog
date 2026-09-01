"""Fine-tune the WTW low-level policy on frequent gait-only transitions.

The original checkpoint and training entry point are never overwritten. During
an added transition event, velocity and all continuous gait parameters remain
unchanged; only the discrete gait family changes.
"""

import argparse
import pickle
from pathlib import Path


DEFAULT_CHECKPOINT = (
    "runs/gait-conditioned-agility/pretrain-go2/train/142238.667503/"
    "checkpoints/ac_weights_019999.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help="Directory containing the parameters.pkl associated with the checkpoint.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--transition-interval", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--impact-scale", type=float, default=-0.01)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def validate_args(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    source_run_dir = (
        Path(args.source_run_dir).expanduser().resolve()
        if args.source_run_dir
        else checkpoint.parent.parent
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not (source_run_dir / "parameters.pkl").is_file():
        raise FileNotFoundError(
            f"Source parameters do not exist: {source_run_dir / 'parameters.pkl'}"
        )
    if output_dir == checkpoint.parent or output_dir in checkpoint.parents:
        raise ValueError("Output directory must not contain or overwrite the source checkpoint")
    if args.iterations <= 0 or args.num_envs <= 0:
        raise ValueError("iterations and num-envs must be positive")
    if args.transition_interval <= 0.0:
        raise ValueError("transition-interval must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.impact_scale > 0.0:
        raise ValueError("impact-scale is a penalty and must be zero or negative")
    return checkpoint, source_run_dir, output_dir


def apply_saved_values(target, values):
    """Restore only declared configuration fields, skipping runtime buffers."""
    for name, value in values.items():
        if hasattr(target, name):
            setattr(target, name, value)


def restore_source_configuration(source_run_dir, Cfg, AC_Args, PPO_Args, RunnerArgs):
    from go2_gym.envs.go2.go2_config import config_go2

    config_go2(Cfg)
    with (source_run_dir / "parameters.pkl").open("rb") as file:
        saved = pickle.load(file)

    for section_name, section_values in saved["Cfg"].items():
        if hasattr(Cfg, section_name) and isinstance(section_values, dict):
            apply_saved_values(getattr(Cfg, section_name), section_values)
    apply_saved_values(AC_Args, saved.get("AC_Args", {}))
    apply_saved_values(PPO_Args, saved.get("PPO_Args", {}))
    apply_saved_values(RunnerArgs, saved.get("RunnerArgs", {}))


def main():
    args = parse_args()
    checkpoint, source_run_dir, output_dir = validate_args(args)

    import isaacgym
    assert isaacgym

    from ml_logger import logger

    from go2_gym.envs.base.legged_robot_config import Cfg
    from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go2_gym_learn.ppo_cse import Runner, RunnerArgs
    from go2_gym_learn.ppo_cse.actor_critic import AC_Args
    from go2_gym_learn.ppo_cse.ppo import PPO_Args
    restore_source_configuration(source_run_dir, Cfg, AC_Args, PPO_Args, RunnerArgs)

    Cfg.env.num_envs = args.num_envs
    Cfg.env.record_video = False
    # Full commands are sampled on reset. Within an episode, the added events
    # change only gait family, so transition attribution remains isolated.
    Cfg.commands.resampling_time = 1000.0
    Cfg.commands.gait_transition_interval_s = args.transition_interval
    Cfg.commands.freeze_curriculum_updates = True
    Cfg.reward_scales.feet_impact_vel = args.impact_scale

    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.adaptation_module_learning_rate = args.learning_rate
    PPO_Args.schedule = "fixed"

    RunnerArgs.resume = False
    RunnerArgs.save_interval = max(1, args.save_interval)
    RunnerArgs.save_video_interval = 0
    RunnerArgs.log_freq = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(output_dir.name, root=output_dir.parent)
    logger.log_text(
        """
charts:
- yKey: train/episode/rew_total/mean
  xKey: iterations
- yKey: train/episode/rew_tracking_lin_vel/mean
  xKey: iterations
- yKey: train/episode/rew_tracking_contacts_shaped_force/mean
  xKey: iterations
- yKey: train/episode/rew_feet_impact_vel/mean
  xKey: iterations
""",
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )

    logger.log_params(
        fine_tune_args=vars(args),
        source_checkpoint=str(checkpoint),
        source_run_dir=str(source_run_dir),
        AC_Args=vars(AC_Args),
        PPO_Args=vars(PPO_Args),
        RunnerArgs=vars(RunnerArgs),
        Cfg=vars(Cfg),
    )

    print("Low-level gait-transition fine-tuning")
    print(f"  source checkpoint: {checkpoint}")
    print(f"  source parameters: {source_run_dir / 'parameters.pkl'}")
    print(f"  output directory:  {output_dir}")
    print(f"  environments:      {args.num_envs}")
    print(f"  iterations:        {args.iterations}")
    print(f"  gait-only interval:{args.transition_interval:.3f}s")
    print(f"  fixed learning rate:{args.learning_rate:g}")
    print(f"  impact reward scale:{args.impact_scale:g}")
    print("  restored source dynamics:")
    print(f"    initial base height: {Cfg.init_state.pos[2]}")
    print(f"    joint stiffness:    {Cfg.control.stiffness}")
    print(f"    joint damping:      {Cfg.control.damping}")
    print(
        f"    terrain tile:      {Cfg.terrain.terrain_length}m x "
        f"{Cfg.terrain.terrain_width}m"
    )

    env = VelocityTrackingEasyEnv(
        sim_device=f"cuda:{args.gpu}",
        headless=not args.render,
        cfg=Cfg,
    )
    env = HistoryWrapper(env)
    runner = Runner(
        env,
        device=f"cuda:{args.gpu}",
        local_checkpoint=str(checkpoint),
    )
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=True,
        eval_freq=100,
    )


if __name__ == "__main__":
    main()
