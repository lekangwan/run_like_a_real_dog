from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_LABEL = "gait-conditioned-agility/pretrain-go2/train"
TASK_MAP = PROJECT_ROOT / "logs/gait_condition_eval_v8_mainline/training_task_map/training_task_map_by_speed.csv"
RUNS_DIR = PROJECT_ROOT / "runs/high_level_oracle_gait"

GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")
RESIDUAL_NAMES = (
    "frequency",
    "duration",
    "footswing_height",
    "stance_width",
    "body_pitch",
)
REWARD_PROFILE = "canonical_efficiency_v4_physical"

HIGH_LEVEL_STEP_SECONDS = 0.1
DEFAULT_DECISION_INTERVAL = 5
MAX_DECISION_INTERVAL = 10

DEFAULT_TASKS = (
    "flat_trot_efficiency",
    "ramp_up_trot_robustness",
)

TERRAIN_LENGTH = 12.0
TERRAIN_WIDTH = 12.0
EDGE_RESET_MARGIN = 1.5
TELEPORT_THRESHOLD = 1.5
MESH_TYPE = "trimesh"


def validate_decision_interval(interval):
    if not 1 <= interval <= MAX_DECISION_INTERVAL:
        raise ValueError(
            "The gait decision interval must be 1-10 high-level steps "
            "(0.1-1.0 seconds). The historical 100-step setting is not allowed."
        )
