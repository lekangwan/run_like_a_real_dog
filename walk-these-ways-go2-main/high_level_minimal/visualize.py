"""Open the same independent evaluator with the Isaac Gym viewer enabled."""

import sys

from .evaluate import main


if __name__ == "__main__":
    if "--render" not in sys.argv:
        sys.argv.append("--render")
    main()
