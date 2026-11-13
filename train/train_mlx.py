"""Maintained Apple Silicon training entry point. Run from the repository root."""

import sys

from prometheus_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
