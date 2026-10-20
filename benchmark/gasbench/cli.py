"""Command-line entry point for GasBench.

    python -m gasbench --help
    python -m gasbench --data ./data --out ./results --families closed --smoke
    python -m gasbench --data ./data --out ./results --families osr,gsa --workers 4

The pipeline is: plan -> execute (cached, hashed) -> aggregate -> freeze.
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .data import ensure_data
from .protocols import task_list, execute_task, aggregate, freeze, PROTOCOL_VERSION
from .utils import SEEDS


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "deploy":
        from .deploy.cli import main as deploy_main
        return deploy_main(argv[1:])
    p = argparse.ArgumentParser(
        prog="gasbench",
        description="GasBench: drift-aware gas sensor recognition benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, type=str,
                   help="data root; raw/ and processed/ are created inside")
    p.add_argument("--out", required=True, type=str,
                   help="results root; tasks/, summary CSVs are written here")
    p.add_argument("--families", default=None, type=_split,
                   help="comma list: closed,budget,forward,singlesource,osr,gsa,gsa_cv,aq (default all)")
    p.add_argument("--seeds", default=None, type=_split,
                   help="comma list of integer seeds (default 42,2024,7,123,2025)")
    p.add_argument("--workers", default=1, type=int,
                   help="parallel processes for task execution")
    p.add_argument("--epochs", default=50, type=int,
                   help="training epochs per method (smoke uses 2)")
    p.add_argument("--smoke", action="store_true",
                   help="run one task per family with 2 epochs (no data download changes)")
    p.add_argument("--aggregate", action="store_true",
                   help="aggregate an already-completed run and exit")
    p.add_argument("--preflight", action="store_true",
                   help="print the plan (task counts per family) and exit")
    p.add_argument("--download-only", action="store_true",
                   help="download/verify data and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {PROTOCOL_VERSION}")
    args = p.parse_args(argv)

    if args.seeds:
        args.seeds = [int(s) for s in args.seeds]
    data_dir = str(Path(args.data).resolve())
    out = str(Path(args.out).resolve())

    if args.download_only:
        proc = ensure_data(data_dir)
        print(f"data ready at {proc}")
        return 0

    if args.preflight:
        _print_plan(args)
        return 0

    if args.aggregate:
        aggregate(out)
        print(f"aggregated {out}")
        return 0

    _print_plan(args)
    if args.smoke:
        _run_smoke(args, data_dir, out)
    else:
        _run(args, data_dir, out)
    return 0


def _print_plan(args):
    tasks = task_list(seeds=args.seeds, families=args.families)
    counts = {}
    for t in tasks:
        counts[t["family"]] = counts.get(t["family"], 0) + 1
    print("PLAN")
    for k, v in sorted(counts.items()):
        print(f"  {k:12s} {v:5d} tasks")
    print(f"  total {len(tasks)} tasks | protocol {PROTOCOL_VERSION} | seeds "
          f"{';'.join(map(str, args.seeds or SEEDS))}")


def _run_smoke(args, data_dir, out):
    """One representative task per family with 2 epochs."""
    ensure_data(data_dir)
    seeds = args.seeds or SEEDS
    tasks = task_list(seeds=[seeds[0]], families=args.families)
    selected, seen = [], set()
    for f in ["closed", "osr", "gsa", "aq", "singlesource"]:
        for t in tasks:
            if t["family"] == f and f not in seen:
                selected.append(t)
                seen.add(f)
                break
    for t in tasks:
        if t["family"] == "gsa_cv" and "gsa_cv" not in seen:
            selected.append(t)
            seen.add("gsa_cv")
        if t["family"] == "forward" and "forward" not in seen:
            selected.append(t)
            seen.add("forward")
        if t["family"] == "budget" and "budget" not in seen:
            selected.append(t)
            seen.add("budget")
    print(f"PLAN smoke tasks={len(selected)} workers=1 smoke=True")
    for t in selected:
        tid, status, secs = execute_task(t, data_dir, out, epochs=2)
        print(f"[smoke] {tid} {status} seconds={secs:.1f}")
    aggregate(out)
    print("FINISHED smoke")


def _run(args, data_dir, out):
    ensure_data(data_dir)
    tasks = task_list(seeds=args.seeds, families=args.families)
    print(f"PLAN tasks={len(tasks)} workers={args.workers} smoke=False")
    if args.workers <= 1:
        for i, t in enumerate(tasks, 1):
            tid, status, secs = execute_task(t, data_dir, out, epochs=args.epochs)
            print(f"[{i}/{len(tasks)}] {tid} {status} seconds={secs:.1f}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(execute_task, t, data_dir, out, args.epochs): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                tid, status, secs = fut.result()
                print(f"[{i}/{len(tasks)}] {tid} {status} seconds={secs:.1f}", flush=True)
    aggregate(out)
    freeze(data_dir, out)
    print("FINISHED")


if __name__ == "__main__":
    sys.exit(main())
