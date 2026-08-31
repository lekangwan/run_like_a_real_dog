"""高层最小实现的集中配置。

模块作用：
    为其余模块提供唯一的默认路径、步态参数语义、地形尺寸和时间尺度定义。

主要输入：
    本文件没有运行时张量输入；路径由当前文件位置推导，其他值是项目主线的固定默认配置。

内部处理：
    1. 定位仓库根目录、底层策略和高层输出目录；
    2. 固定 4 种步态与 5 种连续修正量的索引顺序；
    3. 定义高层环境步长和允许的步态决策周期；
    4. 提供决策周期合法性检查，阻止再次使用不合理的 10 秒决策。

主要输出：
    被 ``tasks/environment/model/train/evaluate/record`` 导入的常量，以及
    ``validate_decision_interval()`` 检查函数。

在整体中的作用：
    它是高层代码的“公共词典”。任何高层指令维度或时间含义发生变化，都应先在这里核对。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_LABEL = "gait-conditioned-agility/pretrain-go2/train"
RUNS_DIR = PROJECT_ROOT / "runs/high_level_oracle_gait"

# 四种离散步态的顺序必须和 WTW 的相位模板、高层步态指令前四维保持一致。
GAIT_NAMES = ("pronking", "trotting", "bounding", "pacing")
# 高层步态指令后五维不是绝对参数，而是相对各步态默认模板的修正量。
RESIDUAL_NAMES = (
    "frequency",
    "duration",
    "footswing_height",
    "stance_width",
    "body_pitch",
)
# 当前底层存档的时间层级（真实来源仍是底层 parameters.pkl）：
#   Isaac Gym 物理步：0.005 秒，200 Hz；执行器网络每个物理步更新一次力矩。
#   WTW 底层策略步：4 个物理步，即 0.020 秒，50 Hz；每次输出 12 维关节目标。
#   高层环境步：5 个底层步，即 0.100 秒，10 Hz；更新历史观测并计算高层奖励。
#   步态决策：默认保持 5 个高层步，即 0.500 秒，2 Hz。
# 这四层不可混为一谈；修改高层决策周期不会改变物理或底层策略频率。
HIGH_LEVEL_STEP_SECONDS = 0.1
# 默认每 5 个高层步重新决策，即 0.5 秒一次。
DEFAULT_DECISION_INTERVAL = 5
# 明确禁止历史实验中的 100 步（10 秒）决策周期。
MAX_DECISION_INTERVAL = 10

DEFAULT_TASKS = (
    "flat_trot_efficiency",
    "ramp_up_trot_robustness",
)

TERRAIN_LENGTH = 12.0
TERRAIN_WIDTH = 12.0
EDGE_RESET_MARGIN = 1.5
TELEPORT_THRESHOLD = 1.5
def validate_decision_interval(interval):
    """检查步态决策周期是否处于 0.1 至 1.0 秒的合理范围。

    输入：
        interval: 整数，一个步态动作需要保持的高层环境步数；每步为 0.1 秒。
    输出：
        无返回值。输入合法时直接结束，非法时抛出 ``ValueError``。
    内部逻辑：
        将步数限制在 1 到 10 之间，从代码层面阻止再次使用历史实验中
        缺乏实时意义的 100 步（10 秒）设置。
    作用：
        统一训练、评测和录像使用的决策时间尺度，防止实验口径失配。
    """
    if not 1 <= interval <= MAX_DECISION_INTERVAL:
        raise ValueError(
            "The gait decision interval must be 1-10 high-level steps "
            "(0.1-1.0 seconds). The historical 100-step setting is not allowed."
        )
