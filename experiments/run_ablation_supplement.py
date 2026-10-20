# -*- coding: utf-8 -*-
"""Figure 4.3 (Sec 4.3) supplementary runs.

Adds methods missing from Figure 4.2:
  - cg_mix     : centroid-guided mixup only (calibration set NOT added to training)
  - cg_shuffle : cg_sup with shuffled (mislabeled) class centroids
  - mean_shift : base model + simple mean-shift calibration

Also extracts penultimate-layer features for b=0.1 tasks and computes:
  - intra_class_distance : mean over classes of ||mu_source_c - mu_target_c||_2
  - inter_class_separation : between-class scatter / within-class scatter (Fisher ratio)

Outputs to:
  REPO/results/main_figure_v2/drift_forward/
  REPO/results/main_figure_v2/cdsl_forward/

Resumable.
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

NEW_METHODS = ['base', 'cg_mix', 'cg_shuffle', 'mean_shift']
BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
FEATURE_BUDGET = 0.10  # extract features only for this budget
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


def compute_feature_metrics(f_source, y_source, f_target_cal, y_target_cal, classes):
    """Compute intra-class distance and inter-class separation in feature space.

    Returns dict with:
      intra_class_distance : mean over classes of ||mu_s - mu_t||_2
      inter_class_separation : between-class scatter / within-class scatter (Fisher ratio)
    """
    results = {}

    # --- intra-class distance: source vs target centroid per class ---
    dists = []
    for c in classes:
        ms = f_source[y_source == c].mean(0) if (y_source == c).sum() > 0 else None
        mt = f_target_cal[y_target_cal == c].mean(0) if (y_target_cal == c).sum() > 0 else None
        if ms is not None and mt is not None:
            dists.append(np.linalg.norm(ms - mt))
    results['intra_class_distance'] = float(np.mean(dists)) if dists else float('nan')

    # --- inter-class separation: between-class scatter / within-class scatter ---
    # Combine all labeled features (source + target calibration)
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


def run_task(task, dataset_name):
    budget, seed, fold = task['budget'], task['seed'], task['fold']
    out = PROJECT / 'results/main_figure_v2' / dataset_name / 'tasks' / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.csv').exists():
        return task['task_id'], 'cached', 0
    t0 = time.time()
    try:
        # Load data
        if dataset_name == 'drift_forward':
            x, y, b = load_drift(str(PROJECT))
            target = np.flatnonzero(b == fold)
            source = np.flatnonzero(b < fold)
            classes = np.arange(1, 7)
        else:  # cdsl_forward
            direction = task['direction']
            x, y, source, target = load_cdsl(direction)
            classes = np.arange(1, 7)

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
        feature_rows = []

        # Train base first (needed for mean_shift)
        t1 = time.time()
        base_model, base_details = train_network(
            xs, y[source], xc, y[cal], 'base', seed, classes, EPOCHS, steps)
        p_base, pr_base, f_base = network_output(base_model, xt, classes)
        r.metric('base', classification_metrics(y[test], p_base, pr_base, classes), n_test=len(test))
        r.predictions('base', test, y[test], p_base)
        r.configrow('base', n_source=len(source), n_target_labeled=0,
                    n_test=len(test), seconds=time.time() - t1, **base_details)

        # mean_shift: reuse base model, apply mean shift
        t1 = time.time()
        shift = xc.mean(0) - xs.mean(0)
        p_ms, pr_ms, f_ms = network_output(base_model, xt - shift, classes)
        r.metric('mean_shift', classification_metrics(y[test], p_ms, pr_ms, classes), n_test=len(test))
        r.predictions('mean_shift', test, y[test], p_ms)
        r.configrow('mean_shift', n_source=len(source), n_target_labeled=len(cal),
                    n_test=len(test), seconds=time.time() - t1, reused_model='base')

        # cg_mix and cg_shuffle
        for m in ['cg_mix', 'cg_shuffle']:
            t1 = time.time()
            model, details = train_network(
                xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            p, pr, f = network_output(model, xt, classes)
            r.metric(m, classification_metrics(y[test], p, pr, classes), n_test=len(test))
            r.predictions(m, test, y[test], p)
            r.configrow(m, n_source=len(source),
                        n_target_labeled=len(cal) if m == 'cg_mix' else len(cal),
                        n_test=len(test), seconds=time.time() - t1, **details)

        # Feature metrics for b=0.1 (panel c)
        if abs(budget - FEATURE_BUDGET) < 1e-6:
            # Get source and target calibration features from base and cg_sup
            # We need cg_sup model too — train it
            t1 = time.time()
            cg_model, _ = train_network(
                xs, y[source], xc, y[cal], 'cg_sup', seed, classes, EPOCHS, steps)
            _, _, f_cg = network_output(cg_model, xt, classes)
            # Also get features for source and calibration
            with __import__('torch').no_grad():
                f_src_base = base_model.fe(__import__('torch').from_numpy(xs)).numpy()
                f_cal_base = base_model.fe(__import__('torch').from_numpy(xc)).numpy()
                f_src_cg = cg_model.fe(__import__('torch').from_numpy(xs)).numpy()
                f_cal_cg = cg_model.fe(__import__('torch').from_numpy(xc)).numpy()

            for method_name, f_s, f_c in [('base', f_src_base, f_cal_base),
                                           ('cg_sup', f_src_cg, f_cal_cg)]:
                fm = compute_feature_metrics(f_s, y[source], f_c, y[cal], classes)
                for k, v in fm.items():
                    feature_rows.append({**task, 'method': method_name, 'metric': k, 'value': v})

        r.finish()
        if feature_rows:
            pd.DataFrame(feature_rows).to_csv(out / 'feature_metrics.csv', index=False)

        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def run_drift(task): return run_task(task, 'drift_forward')
def run_cdsl(task): return run_task(task, 'cdsl_forward')


def build_tasks():
    tasks = []
    for budget in BUDGETS:
        for seed in SEEDS:
            for fold in range(2, 11):
                tasks.append(dict(dataset='UCI_Drift', family='forward', setting='past_to_future',
                                  budget=budget, seed=seed, fold=fold,
                                  task_id=f'drift_forward_b{budget}_s{seed}_f{fold}'))
            for direction in ['master_slave1', 'master_slave2']:
                tasks.append(dict(dataset='CDSL', family='forward', setting='device_cross',
                                  budget=budget, seed=seed, direction=direction,
                                  task_id=f'cdsl_forward_b{budget}_s{seed}_d{direction}'))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  methods={NEW_METHODS}  feature_budget={FEATURE_BUDGET}', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {}
        for t in tasks:
            fn = run_drift if t['dataset'] == 'UCI_Drift' else run_cdsl
            futs[pool.submit(fn, t)] = t
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            print(f'[{i}/{len(tasks)}] {st} {tid} {el:.1f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
