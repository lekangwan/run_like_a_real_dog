from pathlib import Path


MAINLINE_EVAL_DIR = Path("logs/gait_condition_eval_v8_mainline")
MAINLINE_TASK_MAP = MAINLINE_EVAL_DIR / "training_task_map" / "training_task_map_by_speed.csv"
MAINLINE_TEMPLATE_LIBRARY = MAINLINE_EVAL_DIR / "gait_template_library" / "gait_template_library.csv"

VIS_TERRAIN_LENGTH = 30.0
VIS_TERRAIN_WIDTH = 30.0
VIS_TELEPORT_THRESH = 3.0
VIS_EDGE_RESET_MARGIN = 3.0

TRAIN_TERRAIN_SIZE = 12.0
TRAIN_TELEPORT_THRESH = 1.5
TRAIN_EDGE_RESET_MARGIN = 1.5
TRAIN_MESH_TYPE = "trimesh"
