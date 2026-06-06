import sys

from play_push_hard_gait_compare import main


DEFAULT_ARGS = {
    "--condition": "push_longitudinal",
    "--template-condition": "push_hard",
    "--objective": "push_bound_recovery",
    "--gaits": "bounding,trotting",
    "--vx": "1.5",
    "--output-dir": "logs/directed_push_gait_compare",
}


def has_arg(args, name):
    return name in args or any(arg.startswith(f"{name}=") for arg in args)


if __name__ == "__main__":
    user_args = sys.argv[1:]
    injected_args = []
    for name, value in DEFAULT_ARGS.items():
        if not has_arg(user_args, name):
            injected_args.extend([name, value])
    sys.argv = [sys.argv[0]] + injected_args + user_args
    main()
