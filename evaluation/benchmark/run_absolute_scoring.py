"""Compatibility entry point; install the project first (pip install -e .)."""

from prometheus_eval.legacy import legacy_benchmark

if __name__ == "__main__":
    legacy_benchmark("absolute")
