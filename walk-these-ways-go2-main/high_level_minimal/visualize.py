"""打开带 Isaac Gym 可视窗口的独立评测。

本文件不复制评测逻辑，只自动追加 ``--render`` 后复用 evaluate.py。
"""

import sys

from .evaluate import main


if __name__ == "__main__":
    if "--render" not in sys.argv:
        sys.argv.append("--render")
    main()
