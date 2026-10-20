# -*- coding: utf-8 -*-
"""Feature space metrics for correct protocols (LOBO + single-step Forward).

Computes:
  - intra_class_distance : mean ||mu_source_c - mu_target_c||_2 over classes
  - inter_class_separation : between-class scatter / within-class scatter (Fisher ratio)

Methods: base, cal_sup, cg_sup (at b=0.1 only)
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import sys, time, math, traceback
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

METHODS = ['base', 'cal_sup', 'cg_sup', 'cg_shuffle']
BUDGET = 0.01
EPOCHS = 50


def compute_feature_metrics(f_source, y_source, f_target_cal, y_target_cal, classes):
    results = {}
    # intra-class distance: source vs target centroid per class
    dists = []
    for c in classes:
        ms = f_source[y_source == c].mean(0) if (y_source == c).sum() > 0 else None
        mt = f_target_cal[y_target_cal == c].mean(0) if (y_target_cal == c).sum() > 0 else None
        if ms is not None and mt is not None:
            dists.append(np.linalg.norm(ms - mt))
    results['intra_class_distance'] = float(np.mean(dists)) if dists else float('nan')

    # inter-class separation: between / within scatter
    all_f = np.concatenate([f_source, f_target_cal], axis=0)
    all_y = np.concatenate([y_source, y_target_cal], axis=0)
    overall_mean = all_f.mean(0)
    between = 0.0
    within = 0.0
    for c in classes:
        mask = all_y == c
        if mask.sum() < 2:
            continue
        fc = all_f[mask]
        mu_c = fc.mean(0)
        between += mask.sum() * np.sum((mu_c - overall_mean) ** 2)
        within += np.sum((fc - mu_c) ** 2)
    results['inter_class_separation'] = float(between / within) if within > 0 else float('nan')
    return results


def run_task(task):
    budget, seed = task['budget'], task['seed']
    protocol = task['protocol']
    out = PROJECT / 'results/main_figure' / task['out_dir'] / 'tasks' / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.csv').exists():
        return task['task_id'], 'cached', 0
    t0 = time.time()
    try:
        x, y, b = load_drift(str(PROJECT))
        classes = np.arange(1, 7)

        if protocol == 'LOBO':
            source = np.flatnonzero(b == 1)
            target = np.flatnonzero(b == task['target_batch'])
        else:
            source = np.flatnonzero(b == task['source_batch'])
            target = np.flatnonzero(b == task['target_batch'])

        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
        cal = nested_calibration(pool, target, y, budget)
        check_disjoint(source=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 360) / 128)

        rows = []
        centroid_rows = []
        for m in METHODS:
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            # Get features: use penultimate layer
            # network_output returns (preds, probs, features)
            _, _, f_source = network_output(model, xs, classes)
            _, _, f_cal = network_output(model, xc, classes)
            # Compute metrics
            fm = compute_feature_metrics(f_source, y[source], f_cal, y[cal], classes)
            rows.append({
                **task,
                'method': m,
                'intra_class_distance': fm['intra_class_distance'],
                'inter_class_separation': fm['inter_class_separation'],
            })
            # Save per-class centroids
            for c in classes:
                if (y[source] == c).sum() > 0:
                    mu_s = f_source[y[source] == c].mean(0)
                else:
                    mu_s = np.full(f_source.shape[1], np.nan)
                if (y[cal] == c).sum() > 0:
                    mu_t = f_cal[y[cal] == c].mean(0)
                else:
                    mu_t = np.full(f_cal.shape[1], np.nan)
                centroid_rows.append({
                    **task,
                    'method': m,
                    'class': c,
                    'source_centroid': ';'.join(f'{v:.6f}' for v in mu_s),
                    'target_centroid': ';'.join(f'{v:.6f}' for v in mu_t),
                })

        pd.DataFrame(rows).to_csv(out / 'feature_metrics.csv', index=False)
        pd.DataFrame(centroid_rows).to_csv(out / 'centroids.csv', index=False)
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks():
    tasks = []
    for seed in SEEDS:
        for tgt in range(2, 11):
            tasks.append(dict(
                protocol='LOBO', out_dir='feature_lobo_correct',
                budget=BUDGET, seed=seed,
                source_batch=1, target_batch=tgt,
                task_id=f'lobo_b{BUDGET}_s{seed}_1to{tgt}'
            ))
        for k in range(1, 10):
            tasks.append(dict(
                protocol='Forward', out_dir='feature_fwd_single',
                budget=BUDGET, seed=seed,
                source_batch=k, target_batch=k+1,
                task_id=f'fwd_b{BUDGET}_s{seed}_{k}to{k+1}'
            ))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  (3 methods, b=0.1)', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            print(f'[{i}/{len(tasks)}] {st} {tid} {el:.1f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
