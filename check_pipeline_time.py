"""Simple helper to measure pipeline runtime locally.
Not used by the checker, only for developer benchmarking.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from pipeline import run_pipeline
from api_state import ApiState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure pipeline runtime on two data versions")
    p.add_argument("--data-v1", type=Path, required=True)
    p.add_argument("--data-v2", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = ApiState()

    t0 = time.perf_counter()
    run_pipeline(version="v1", state=state, data_root=args.data_v1)
    t1 = time.perf_counter()
    print(f"v1 pipeline done in {t1 - t0:.2f}s")

    t2 = time.perf_counter()
    run_pipeline(version="v2", state=state, data_root=args.data_v2)
    t3 = time.perf_counter()
    print(f"v2 pipeline done in {t3 - t2:.2f}s")
    print(f"total: {t3 - t0:.2f}s")


if __name__ == "__main__":
    main()
