"""Minimal library usage example.

Run: python examples/quickstart.py --data ../data --out ../results
"""
import argparse
import sys
from pathlib import Path

import gasbench as gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "results"))
    args = ap.parse_args()

    # 1) Data: downloads if missing.
    x, y, batch = gb.load_drift(args.data)
    print(f"drift: {x.shape}, {len(set(y))} classes, batches {sorted(set(batch))}")

    # 2) A tiny single-source experiment, one seed, one target batch.
    from gasbench.utils import SEEDS, check_disjoint, stratified_parts, nested_calibration
    from gasbench.protocols import execute_task

    task = {"family": "singlesource", "setting": "b1", "budget": 0.1,
            "seed": SEEDS[0], "fold": 8}
    tid, status, seconds = execute_task(task, args.data, args.out, epochs=2)
    print(f"task {tid}: {status} ({seconds:.1f}s, smoke epochs)")

    # 3) Aggregate everything collected so far.
    from gasbench.protocols import aggregate
    aggregate(args.out)
    print("done — see", Path(args.out) / "summary_mean_sd.csv")


if __name__ == "__main__":
    sys.exit(main())
