# -*- coding: utf-8 -*-
"""CDSL sensor-drift full experiments: 6 methods x 5 budgets x 2 directions x 5 seeds.

Directions: master -> slave1, master -> slave2
Methods:    base, cal_sup, mixup_std, cg_sup, cg_mix, cg_shuffle
Budgets:    0.01, 0.05, 0.10, 0.15, 0.20
Seeds:      [7, 42, 123, 2024, 2025]

Outputs (main_results/):
  table_cdsl_full_macro_f1.csv   wide mean/std (direction x budget x 6 methods), macro_f1
  table_cdsl_full_accuracy.csv   same layout, accuracy
  table_cdsl_full_long.csv       long (direction, budget, seed, method, metric, value)
  table_cdsl_full_means.csv      long means over seeds (direction, budget, method, metric, value, std)

Per-task raw metrics: results/cdsl_full/tasks/<task_id>/metrics.csv
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
from run_revision import shuffled_pool

DATA_DIR = PROJECT / 'data' / 'CDSL_csv'
OUT_DIR = PROJECT / 'main_results'
TASKS_DIR = PROJECT / 'results' / 'cdsl_full' / 'tasks'

METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle']
BUDGETS = [0.01, 0.05, 0.10, 0.15, 0.20]
EPOCHS = 50
DIRECTIONS = ['master_slave1', 'master_slave2']


def load_cdsl(direction):
    fname = 'cdsl_master_slave1.csv' if direction == 'master_slave1' else 'cdsl_master_slave2.csv'
    df = pd.read_csv(DATA_DIR / fname)
    x = df[[f'f{i}' for i in range(1, 7)]].values.astype(np.float64)
    y = df['label'].values.astype(int)
    dom = df['domain'].values
    src = np.flatnonzero(dom == 'master')
    tgt = np.flatnonzero(dom != 'master')
    return x, y, src, tgt


def run_task(task):
    direction, budget, seed = task['direction'], task['budget'], task['seed']
    out = TASKS_DIR / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion.csv').exists():
        return task['task_id'], 'cached', 0
    t0 = time.time()
    try:
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

        rows = []
        for m in METHODS:
            t1 = time.time()
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            p, pr, _ = network_output(model, xt, classes)
            metrics = classification_metrics(y[test], p, pr, classes)
            n_params = details.get('parameter_count', sum(p.numel() for p in model.parameters()))
            for k, v in metrics.items():
                rows.append({'direction': direction, 'budget': budget, 'seed': seed,
                             'method': m, 'metric': k, 'value': float(v),
                             'n_source': int(len(source)), 'n_cal': int(len(cal)), 'n_test': int(len(test)),
                             'n_params': int(n_params), 'train_s': round(time.time() - t1, 3)})
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False, float_format='%.12g')
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks():
    tasks = []
    for direction in DIRECTIONS:
        for budget in BUDGETS:
            for seed in SEEDS:
                tasks.append(dict(direction=direction, budget=budget, seed=seed,
                                  task_id=f'cdsl_{direction}_b{budget}_s{seed}'))
    return tasks


def aggregate():
    rows = []
    for d in DIRECTIONS:
        for b in BUDGETS:
            for s in SEEDS:
                f = TASKS_DIR / f'cdsl_{d}_b{b}_s{s}' / 'metrics.csv'
                if f.exists():
                    rows.append(pd.read_csv(f))
    if not rows:
        print('no results yet', flush=True); return
    df = pd.concat(rows, ignore_index=True)

    # long per-seed
    long = df[['direction', 'budget', 'seed', 'method', 'metric', 'value',
               'n_source', 'n_cal', 'n_test', 'n_params', 'train_s']]
    long.to_csv(OUT_DIR / 'table_cdsl_full_long.csv', index=False, float_format='%.12g')

    # means + std over seeds (long)
    g = df.groupby(['direction', 'budget', 'method', 'metric'], as_index=False)['value']
    means = g.mean().rename(columns={'value': 'value_mean'})
    stds = g.std().rename(columns={'value': 'value_std'})
    msum = means.merge(stds, on=['direction', 'budget', 'method', 'metric'])
    msum.to_csv(OUT_DIR / 'table_cdsl_full_means.csv', index=False, float_format='%.12g')

    # wide macro_f1 (direction x budget x methods)
    f1 = msum[msum['metric'] == 'macro_f1']
    wf1 = f1.pivot_table(index=['direction', 'budget'], columns='method',
                         values='value_mean').reset_index()
    wf1.to_csv(OUT_DIR / 'table_cdsl_full_macro_f1.csv', index=False, float_format='%.6f')

    acc = msum[msum['metric'] == 'accuracy']
    wa = acc.pivot_table(index=['direction', 'budget'], columns='method',
                         values='value_mean').reset_index()
    wa.to_csv(OUT_DIR / 'table_cdsl_full_accuracy.csv', index=False, float_format='%.6f')

    print('=== macro_f1 (mean over 5 seeds) ===', flush=True)
    print(wf1.to_string(index=False), flush=True)
    print(f'saved 4 files -> {OUT_DIR}', flush=True)


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  ({len(DIRECTIONS)} dir x {len(BUDGETS)} budget x {len(SEEDS)} seed x {len(METHODS)} method)', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            if i % 5 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] done={done} fail={fail} wall={time.time()-t0:.0f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)
    aggregate()


if __name__ == '__main__':
    main()
