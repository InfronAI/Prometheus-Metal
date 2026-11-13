"""Local replacement for the original CUDA-only inference example."""

import sys
from prometheus_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
