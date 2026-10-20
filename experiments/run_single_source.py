# -*- coding: utf-8 -*-
"""Single-source multi-target experiments: source = batch k-1, target = batch k..10.

k from 3 to 9.
Methods: base, cal_sup, mixup_std, cg_sup, cg_mix, cg_shuffle, cg_minus_cal, cg_minus_base
Budgets: 0.01, 0.05, 0.1, 0.15, 0.2
Seeds: 5
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import sys, time, math, traceback, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

PROJECT = REPO
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))

from revision_core import (SEEDS, StandardScaler, stratified_parts,
                           nested_calibration, check_disjoint,
                           train_network, network_output, classification_metrics)
from run_revision import load_drift, shuffled_pool

EPOCHS = 50

METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle', 'cg_minus_cal', 'cg_minus_base']
BUDGETS = [0.01, 0.05, 0.1, 0.15, 0.2]
SEEDS = [7, 42, 123, 2024, 2025]


def run_task(task):
    method = task['method']
    budget = task['budget']
    source_batch = task['source_batch']
    target_batch = task['target_batch']
    seed = task['seed']

    out_dir = PROJECT / 'results/single_source' / method / f'b{budget}' / f's{source_batch}_t{target_batch}_seed{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / 'completion.csv').exists():
        return task['task_id'], 'cached', 0

    t0 = time.time()
    try:
        x, y, b = load_drift(str(PROJECT))
        classes = np.arange(1, 7)

        source = np.flatnonzero(b == source_batch)
        target = np.flatnonzero(b == target_batch)

        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
        cal = nested_calibration(pool, target, y, budget)
        check_disjoint(source=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 360) / 128)

        model, details = train_network(xs, y[source], xc, y[cal], method, seed, classes, EPOCHS, steps)
        logits, probs, features = network_output(model, xt, classes)
        metrics = classification_metrics(y[test], logits, probs, classes)

        pd.DataFrame([{
            'source_batch': source_batch,
            'target_batch': target_batch,
            'direction': f'{source_batch}to{target_batch}',
            'budget': budget,
            'seed': seed,
            'method': method,
            'macro_f1': metrics['macro_f1'],
            'accuracy': metrics['accuracy'],
        }]).to_csv(out_dir / 'metrics.csv', index=False)

        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)

        return task['task_id'], 'complete', time.time() - t0

    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks():
    tasks = []
    for k in range(2, 11):  # k from 2 to 10
        source_batch = k - 1
        for target_batch in range(k, 11):
            for method in METHODS:
                for budget in BUDGETS:
                    for seed in SEEDS:
                        tasks.append(dict(
                            method=method,
                            budget=budget,
                            source_batch=source_batch,
                            target_batch=target_batch,
                            seed=seed,
                            task_id=f's{source_batch}_t{target_batch}_m{method}_b{budget}_s{seed}'
                        ))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            if i % 50 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] done={done}, fail={fail}, elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
