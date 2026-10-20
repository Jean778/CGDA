# -*- coding: utf-8 -*-
"""Supplemental run: mixup_std under the forward_past_to_future protocol.

Why a separate script: run_revision.py pins the forward protocol to
['base','cal_sup','cg_sup']. To avoid overwriting existing metrics.csv
(Recorder.finish rewrites it), this script:
  1. reuses splits.csv + preprocessing.csv from each forward task directory;
  2. trains only mixup_std;
  3. appends mixup_std metrics to the existing metrics.csv and writes predictions_mixup_std.csv.
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os, sys, glob, re, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
from revision_core import train_network, network_output, classification_metrics

PROJECT = str(REPO)
TASKS_ROOT = str(REPO / 'results' / 'revision_v2' / 'tasks')
EPOCHS = 50
CLASSES = np.arange(1, 7)


def load_drift():
    d = pd.read_csv(PROJECT + '/data/processed/drift_clean.csv')
    x = np.log1p(np.maximum(d.filter(regex='^s').values, 0)).astype(np.float32)
    return x, d.gas_id.values


def run_one(task_dir):
    name = os.path.basename(task_dir)
    m = re.match(r'forward_past_to_future_b0\.1_s(\d+)_f(\d+)', name)
    if not m:
        return name, 'skip-not-forward', 0
    seed, fold = int(m.group(1)), int(m.group(2))

    done_flag = os.path.join(task_dir, 'predictions_mixup_std.csv')
    if os.path.exists(done_flag):
        return name, 'cached', 0

    t0 = time.time()
    splits = pd.read_csv(os.path.join(task_dir, 'splits.csv'))
    source = splits.loc[splits.role == 'source_train', 'sample_id'].values
    cal = splits.loc[splits.role == 'calibration_train', 'sample_id'].values
    test = splits.loc[splits.role == 'test', 'sample_id'].values
    pool = splits.loc[splits.role == 'unused_pool', 'sample_id'].values

    prep = pd.read_csv(os.path.join(task_dir, 'preprocessing.csv'))
    mean = prep['mean'].values.astype(np.float32)
    scale = prep['scale'].values.astype(np.float32)

    x, y = load_drift()
    # preprocessing.csv mean/scale were fit on raw (log1p) source features
    xs = ((x[source] - mean) / scale).astype(np.float32)
    xc = ((x[cal] - mean) / scale).astype(np.float32)
    xt = ((x[test] - mean) / scale).astype(np.float32)

    steps = int(np.ceil((len(source) + len(pool) + 360) / 128))
    model, details = train_network(xs, y[source], xc, y[cal], 'mixup_std',
                                   seed, CLASSES, EPOCHS, steps)
    p, pr, _ = network_output(model, xt, CLASSES)
    metrics = classification_metrics(y[test], p, pr, CLASSES)

    # append to metrics.csv (keep existing base/cal_sup/cg_sup rows)
    metrics_csv = os.path.join(task_dir, 'metrics.csv')
    old = pd.read_csv(metrics_csv)
    new_rows = []
    for k, v in metrics.items():
        new_rows.append({
            'family': 'forward', 'setting': 'past_to_future', 'budget': 0.1,
            'seed': seed, 'fold': fold, 'method': 'mixup_std',
            'threshold_rule': 'none', 'evaluation': 'heldout',
            'metric': k, 'value': float(v), 'n_test': len(test)})
    pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True).to_csv(
        metrics_csv, index=False, float_format='%.12g')

    # predictions
    pd.DataFrame({'sample_id': test, 'y_true': y[test], 'y_pred': p}).to_csv(
        done_flag, index=False)

    return name, 'complete', time.time() - t0


def main():
    dirs = sorted(glob.glob(os.path.join(TASKS_ROOT, 'forward_past_to_future_b0.1_s*_f*')))
    print(f'PLAN forward mixup_std tasks={len(dirs)}', flush=True)
    t0 = time.time()
    for i, d in enumerate(dirs, 1):
        try:
            tid, st, el = run_one(d)
            print(f'[{i}/{len(dirs)}] {st} {tid} {el:.1f}s', flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[FAIL] {d}: {e!r}', flush=True)
    print(f'FINISHED in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
