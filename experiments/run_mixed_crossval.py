"""Stratified mixed-batch/window cross-validation (diagnostic control).

Panel a needs mixed CV as a diagnostic control: train and test mix across
all batches/windows, reproducing no domain shift, showing only recognition
performance on mixed data. Runs base only, compared with closed/forward base.

Output is isomorphic to summary_mean_sd.csv from run_revision:
    family=mixedcv, setting=mixed_stratified_cv5, budget=0, method=base,
    threshold_rule=none, evaluation=heldout, metric=..., mean, std, n_seeds
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import hashlib, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from revision_core import (SEEDS, classification_metrics, network_output,
                           seed_summary, train_network, check_disjoint)

VERSION = 'mixedcv_v1_20260920'
KEYS = ['family', 'setting', 'budget', 'method', 'threshold_rule', 'evaluation']
PROJECT = REPO
OUT_ROOT = PROJECT / 'results' / 'mixedcv_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(df); temp = path.with_suffix('.csv.tmp')
    df.to_csv(temp, index=False, float_format='%.12g'); temp.replace(path)


def load_drift():
    d = pd.read_csv(PROJECT / 'data/processed/drift_clean.csv')
    x = np.log1p(np.maximum(d.filter(regex='^s').values, 0)).astype(np.float32)
    return x, d.gas_id.values, np.arange(1, 7)  # classes 1..6


def task_list():
    tasks = []
    for ds in ['drift']:  # CDSL has no time windows; mixed CV is UCI-only
        for seed in SEEDS:
            for fold in range(5):
                tasks.append(dict(dataset=ds, seed=seed, fold=fold))
    return tasks


def task_id(t): return f"{t['dataset']}_mixedcv_s{t['seed']}_f{t['fold']}"


def execute(t, epochs=50):
    out = OUT_ROOT / 'tasks' / task_id(t); out.mkdir(parents=True, exist_ok=True)
    done = out / 'completion.csv'
    if done.exists(): return task_id(t), 'cached', 0.0
    started = time.time()
    ds = t['dataset']; seed = t['seed']; fold = t['fold']
    try:
        x, y, classes = load_drift()
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        splits = list(skf.split(x, y))
        train_idx, test_idx = splits[fold]
        sc = StandardScaler().fit(x[train_idx])
        xs = sc.transform(x[train_idx]).astype(np.float32)
        xt = sc.transform(x[test_idx]).astype(np.float32)
        xc = np.empty((0, xs.shape[1]), dtype=np.float32)
        yc = np.empty((0,), dtype=int)
        steps = math_ceil(len(xs) / 128)
        model, details = train_network(xs, y[train_idx], xc, yc, 'base', seed,
                                       classes, epochs=epochs, steps=steps)
        p, pr, _ = network_output(model, xt, classes)
        metrics = classification_metrics(y[test_idx], p, pr, classes)
        rows = [{'family': 'mixedcv', 'setting': 'mixed_stratified_cv5', 'budget': 0,
                 'method': 'base', 'threshold_rule': 'none', 'evaluation': 'heldout',
                 'dataset': ds, 'seed': seed, 'fold': fold, 'metric': k, 'value': float(v),
                 'n_test': len(test_idx)} for k, v in metrics.items()]
        csv(rows, out / 'metrics.csv')
        csv({'sample_id': test_idx, 'y_true': y[test_idx], 'y_pred': p},
            out / 'predictions_base.csv')
        csv([{'status': 'complete', 'seconds': time.time() - started}], done)
        return task_id(t), 'complete', time.time() - started
    except Exception as e:
        traceback.print_exc()
        csv([{'status': 'failed', 'error': repr(e), 'seconds': time.time() - started}],
            out / 'failure.csv')
        return task_id(t), 'failed', time.time() - started


def math_ceil(n):
    return int(np.ceil(max(n, 1)))


def aggregate():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frames = []
    status = []
    for t in task_list():
        p = OUT_ROOT / 'tasks' / task_id(t)
        done = (p / 'completion.csv').exists()
        status.append({**t, 'task_id': task_id(t),
                       'status': 'complete' if done else 'failed'})
        if done:
            frames.append(pd.read_csv(p / 'metrics.csv'))
    csv(status, OUT_ROOT / 'task_status.csv')
    data = pd.concat(frames, ignore_index=True)
    csv(data, OUT_ROOT / 'metrics_by_fold_seed.csv')
    per, agg = seed_summary(data, KEYS + ['dataset'])
    csv(per, OUT_ROOT / 'metrics_by_seed.csv')
    csv(agg, OUT_ROOT / 'summary_mean_sd.csv')
    _, fold = seed_summary(data, KEYS + ['dataset', 'fold'])
    csv(fold, OUT_ROOT / 'summary_by_fold_mean_sd.csv')
    print('\n=== mixedcv summary ===')
    print(agg[agg.metric == 'accuracy'].to_string(index=False))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = task_list()
    print(f'PLAN mixedcv tasks={len(tasks)}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(execute, t): t for t in tasks}
        for n, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            print(f'[{n}/{len(tasks)}] {st} {tid} {el:.1f}s', flush=True)
    aggregate()
    print('FINISHED', flush=True)


if __name__ == '__main__':
    main()
