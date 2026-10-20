# -*- coding: utf-8 -*-
"""Figure 1 (Sec 4.2) supplementary runs.

Runs Forward protocol on UCI Drift (A) and CDSL (B) with:
  - 5 label budgets: 1%, 5%, 10%, 15%, 20%
  - 4 methods: base, cal_sup, mixup_std, cg_sup
  - 5 seeds x folds (UCI: 9 folds f2..f10; CDSL: 2 directions master->slave1/2)

C (GSA MixedCV) already exists in results/revision_v2, no rerun needed.

Outputs to:
  REPO/results/main_figure/drift_forward/
  REPO/results/main_figure/cdsl_forward/

Resumable: a task with completion.csv is skipped.
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
from run_cdsl_full import load_cdsl

TARGET_METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup']
BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
EPOCHS = 50


# ---------- shared recorder ----------
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
    def split(self, **parts):
        rows = [{'role': k, 'sample_id': int(i)} for k, z in parts.items() for i in z]
        pd.DataFrame(rows).to_csv(self.out / 'splits.csv', index=False)
    def finish(self):
        pd.DataFrame(self.metrics).to_csv(self.out / 'metrics.csv', index=False, float_format='%.12g')
        pd.DataFrame(self.config).to_csv(self.out / 'model_details.csv', index=False)


# ---------- UCI Drift forward ----------
def run_drift_forward(task):
    budget, seed, fold = task['budget'], task['seed'], task['fold']
    out = PROJECT / 'results/main_figure/drift_forward/tasks' / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.csv').exists():
        return task['task_id'], 'cached', 0
    t0 = time.time()
    try:
        x, y, b = load_drift(str(PROJECT))
        target = np.flatnonzero(b == fold)
        source = np.flatnonzero(b < fold)
        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
        cal = nested_calibration(pool, target, y, budget)
        check_disjoint(source=source, calibration=cal, test=test)
        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 360) / 128)
        classes = np.arange(1, 7)
        r = Recorder(task, out)
        r.split(source_train=source, calibration_train=cal, test=test,
                unused_pool=np.setdiff1d(pool, cal))
        pd.DataFrame({'feature': np.arange(x.shape[1]), 'mean': sc.mean_, 'scale': sc.scale_}).to_csv(
            out / 'preprocessing.csv', index=False)
        base = None
        for m in TARGET_METHODS:
            t1 = time.time()
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            p, pr, _ = network_output(model, xt, classes)
            r.metric(m, classification_metrics(y[test], p, pr, classes), n_test=len(test))
            r.predictions(m, test, y[test], p)
            r.configrow(m, n_source=len(source),
                        n_target_labeled=len(cal) if m in ('cal_sup', 'cg_sup') else 0,
                        n_test=len(test), seconds=time.time() - t1, **details)
        r.finish()
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


# ---------- CDSL forward ----------
def run_cdsl_forward(task):
    budget, seed, direction = task['budget'], task['seed'], task['direction']
    out = PROJECT / 'results/main_figure/cdsl_forward/tasks' / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.csv').exists():
        return task['task_id'], 'cached', 0
    t0 = time.time()
    try:
        x, y, source, target = load_cdsl(direction)
        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
        cal = nested_calibration(pool, target, y, budget)
        check_disjoint(source=source, calibration=cal, test=test)
        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 360) / 128)
        classes = np.arange(1, 7)
        r = Recorder(task, out)
        r.split(source_train=source, calibration_train=cal, test=test,
                unused_pool=np.setdiff1d(pool, cal))
        pd.DataFrame({'feature': np.arange(x.shape[1]), 'mean': sc.mean_, 'scale': sc.scale_}).to_csv(
            out / 'preprocessing.csv', index=False)
        base = None
        for m in TARGET_METHODS:
            t1 = time.time()
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            p, pr, _ = network_output(model, xt, classes)
            r.metric(m, classification_metrics(y[test], p, pr, classes), n_test=len(test))
            r.predictions(m, test, y[test], p)
            r.configrow(m, n_source=len(source),
                        n_target_labeled=len(cal) if m in ('cal_sup', 'cg_sup') else 0,
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
            for fold in range(2, 11):  # UCI Drift f2..f10
                tasks.append(dict(dataset='UCI_Drift', family='forward', setting='past_to_future',
                                  budget=budget, seed=seed, fold=fold,
                                  task_id=f'drift_forward_b{budget}_s{seed}_f{fold}'))
            for direction in ['master_slave1', 'master_slave2']:  # CDSL device-cross
                tasks.append(dict(dataset='CDSL', family='forward', setting='device_cross',
                                  budget=budget, seed=seed, direction=direction,
                                  task_id=f'cdsl_forward_b{budget}_s{seed}_d{direction}'))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  budgets={BUDGETS}  methods={TARGET_METHODS}', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {}
        for t in tasks:
            fn = run_drift_forward if t['dataset'] == 'UCI_Drift' else run_cdsl_forward
            futs[pool.submit(fn, t)] = t
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            print(f'[{i}/{len(tasks)}] {st} {tid} {el:.1f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
