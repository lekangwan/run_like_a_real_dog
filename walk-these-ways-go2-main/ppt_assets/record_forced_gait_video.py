#!/usr/bin/env python3
"""Record one explicitly forced gait using the existing project recorder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import record_high_level_policy_videos as recorder
import torch
from train_high_level_oracle_ppo import GAIT_NAMES


def main() -> None:
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--force-gait", required=True, choices=GAIT_NAMES)
    wrapper_args, recorder_args = wrapper_parser.parse_known_args()

    metadata_parser = argparse.ArgumentParser(add_help=False)
    metadata_parser.add_argument("--eval", required=True)
    metadata_parser.add_argument("--output-dir", required=True)
    metadata, _ = metadata_parser.parse_known_args(recorder_args)

    original_load_model = recorder.load_model

    def load_forced_model(checkpoint, env, run_args):
        model, iteration = original_load_model(checkpoint, env, run_args)
        gait_id = GAIT_NAMES.index(wrapper_args.force_gait)

        def forced_action(obs):
            action = torch.zeros(
                obs.shape[0], env.num_high_level_actions,
                device=obs.device, dtype=obs.dtype,
            )
            action[:, gait_id] = 1.0
            return action

        model.act_student_selector_only = forced_action
        model.act_student = forced_action
        return model, iteration

    recorder.load_model = load_forced_model
    sys.argv = [sys.argv[0], *recorder_args]
    recorder.main()

    task_id, vx = recorder.parse_single_eval(metadata.eval)
    json_path = Path(metadata.output_dir) / f"{recorder.video_stem(task_id, vx)}.json"
    payload = json.loads(json_path.read_text())
    payload["control_mode"] = "forced_gait_visualization"
    payload["forced_gait"] = wrapper_args.force_gait
    payload["claim_limit"] = "Visualizes gait execution; not evidence of autonomous gait selection."
    json_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
