# -*- coding: utf-8 -*-
"""Correct protocol settings for Sec 4.2 summary.

LOBO: source = batch 1 (fixed), target = batch 2..10 (9 directions)
Forward: single-step adjacent (1->2, 2->3, ..., 9->10) (9 directions)

Outputs to:
  REPO/results/main_figure/drift_lobo_correct/
  REPO/results/main_figure/drift_fwd_single/
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
from run_revision import load_drift, csv, shuffled_pool

TARGET_METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle']
BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
EPOCHS = 50


class Recorder:
    def __init__(self, task, out):
        self.t = task; self.out = Path(out); self.metrics = []; self.config = []
    def metric(self, method, values, threshold_rule='none', evaluation='heldout', **extras):
        for k, v in values.items():
            self.metrics.append({**self.t, 'method': method, 'threshold_rule': threshold_rule,
                                 'evaluation': evaluation, 'metric': k, 'value': float(v), **extras})
    def configrow(self, method, **kwargs):
        self.config.append({**self.t, 'method': method, **kwargs})
    def predictions(self, method, ids, y, p):
        pd.DataFrame({'sample_id': ids, 'y_true': y, 'y_pred': p}).to_csv(
            self.out / f'predictions_{method}.csv', index=False)
    def finish(self):
        pd.DataFrame(self.metrics).to_csv(self.out / 'metrics.csv', index=False, float_format='%.12g')
        pd.DataFrame(self.config).to_csv(self.out / 'model_details.csv', index=False)


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
            # source = batch 1 (fixed), target = batch A
            source = np.flatnonzero(b == 1)
            target = np.flatnonzero(b == task['target_batch'])
        else:  # Forward single-step
            # source = batch k, target = batch k+1
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

        r = Recorder(task, out)
        for m in TARGET_METHODS:
            t1 = time.time()
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            p, pr, _ = network_output(model, xt, classes)
            r.metric(m, classification_metrics(y[test], p, pr, classes),
                     n_test=len(test), n_source=len(source), n_cal=len(cal))
            r.configrow(m, n_source=len(source), n_target_labeled=len(cal),
                        n_test=len(test), seconds=time.time() - t1, **details)
        r.finish()
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks():
    tasks = []
    for budget in BUDGETS:
        for seed in SEEDS:
            # LOBO: source=batch1, target=batch2..10
            for tgt in range(2, 11):
                tasks.append(dict(
                    protocol='LOBO', out_dir='drift_lobo_correct',
                    budget=budget, seed=seed,
                    source_batch=1, target_batch=tgt,
                    task_id=f'lobo_b{budget}_s{seed}_1to{tgt}'
                ))
            # Forward single-step: source=k, target=k+1, k=1..9
            for k in range(1, 10):
                tasks.append(dict(
                    protocol='Forward', out_dir='drift_fwd_single',
                    budget=budget, seed=seed,
                    source_batch=k, target_batch=k+1,
                    task_id=f'fwd_b{budget}_s{seed}_{k}to{k+1}'
                ))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  (2 protocols x 9 directions x 5 budget x 5 seed)', flush=True)
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
