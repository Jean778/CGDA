# -*- coding: utf-8 -*-
"""Measure CGDA (cg_sup) metrics on the same fixed test sets as literature models.

A: UCI Drift  source=batch1 -> target=batch2 (1to2), budget=0.1
B: CDSL source=master -> target=slave1, budget=0.1
C: GSA       target=50ppb, source=100+200 (LOCO),     budget=0.02

Per (dataset) runs in one fresh subprocess; parent polls RSS peak via psutil.
Outputs macro_f1, accuracy, ece, inference_ms, peak_rss_MB, training_s.

Output: main_results/table_46_cgda_fixed_test.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '6'
import sys, math, time, subprocess, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

PROJECT = REPO
OUT = PROJECT / 'main_results'
SEED = 7
EPOCHS = 50
DATASETS = [('A', 'UCI_Drift', 0.1), ('B', 'CDSL', 0.1), ('C', 'GSA', 0.02)]

CHILD = textwrap.dedent(r'''
    import os, sys, math
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '6'
    sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
    sys.path.insert(0, str(REPO / 'experiments'))
sys.path.insert(0, str(REPO / 'overhead'))
    import numpy as np
    import torch
    from revision_core import (StandardScaler, stratified_parts, nested_calibration,
                               check_disjoint, train_network, network_output,
                               classification_metrics, ece)
    from run_revision import load_drift, shuffled_pool
    from run_cdsl_full import load_cdsl
    import time
    EPOCHS = 50
    dataset = sys.argv[1]
    budget = float(sys.argv[2])
    seed = int(sys.argv[3])
    if dataset == 'A':
        x, y, b = load_drift(str(REPO))
        source = np.flatnonzero(b == 1); target = np.flatnonzero(b == 2)
        classes = np.arange(1, 7)
        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
    elif dataset == 'B':
        x, y, source, target = load_cdsl('master_slave1')
        classes = np.arange(1, 7)
        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
    else:
        from run_gsa_loco import load_gsa
        x, y, conc = load_gsa(str(REPO)); y = y + 1
        source = np.flatnonzero(conc != 50); target = np.flatnonzero(conc == 50)
        classes = np.arange(1, 7)
        # same split style as run_gsa_loco: per-class cal from target, rest test
        rng = np.random.RandomState(seed)
        cal_ids = []
        for c in np.unique(y[target]):
            z = target[y[target] == c]
            rng.shuffle(z)
            n = min(len(z), max(1, int(budget * np.sum(y[target] == c))))
            cal_ids.extend(z[:n])
        cal = np.array(sorted(cal_ids))
        test = np.array(sorted(np.setdiff1d(target, cal)))
        check_disjoint(source_train=source, calibration=cal, test=test)
        # replicate gsa train path directly (no pool concept)
        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(cal) + 360) / 128)
        model, details = train_network(xs, y[source], xc, y[cal], 'cg_sup', seed, classes, EPOCHS, steps)
        logits, probs, features = network_output(model, xt, classes)
        m = classification_metrics(y[test], logits, probs, classes)
        ecev = ece(probs, y[test], classes)
        # latency
        model.eval()
        xxt = torch.from_numpy(xt.reshape(len(xt), 1, -1))
        with torch.no_grad():
            _ = model(xxt[:16]); t0 = time.time_ns()
            _ = model(xxt); t1 = time.time_ns()
        lat = (t1 - t0) / 1e6 / len(xt)
        print('RESULT', m['macro_f1'], m['accuracy'], ecev, lat, details.get('parameter_count', sum(p.numel() for p in model.parameters())))
        raise SystemExit(0)
    cal = nested_calibration(pool, target, y, budget)
    check_disjoint(source=source, calibration=cal, test=test)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xc = sc.transform(x[cal]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    model, details = train_network(xs, y[source], xc, y[cal], 'cg_sup', seed, classes, EPOCHS, steps)
    logits, probs, features = network_output(model, xt, classes)
    m = classification_metrics(y[test], logits, probs, classes)
    ecev = ece(probs, y[test], classes)
    model.eval()
    xxt = torch.from_numpy(xt.reshape(len(xt), 1, -1))
    with torch.no_grad():
        for _ in range(5):
            _ = model(xxt[:16])
        t0 = time.perf_counter()
        _ = model(xxt)
        t1 = time.perf_counter()
    lat = (t1 - t0) * 1000.0 / len(xt)  # ms per sample, batch-amortized (same as lit models)
    print('RESULT', m['macro_f1'], m['accuracy'], ecev, lat, details.get('parameter_count', sum(p.numel() for p in model.parameters())))
''')


def run_sub(dataset, budget):
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, '-c', CHILD, dataset, str(budget), str(SEED)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    peak = 0
    while True:
        try:
            peak = max(peak, psutil.Process(proc.pid).memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if proc.poll() is not None:
            try:
                peak = max(peak, psutil.Process(proc.pid).memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            break
        time.sleep(0.05)
    out, err = proc.communicate()
    if proc.returncode != 0:
        return None, err.decode(errors='ignore')[-600:]
    for line in out.decode(errors='ignore').splitlines():
        if line.startswith('RESULT'):
            parts = line.split()[1:]
            return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])), peak
    return None, 'no RESULT line'


def main():
    rows = []
    for ds_label, ds_name, budget in DATASETS:
        t0 = time.time()
        res, peak = run_sub(ds_label, budget)
        if res is None:
            print(f'{ds_label}: FAILED {peak}', flush=True)
            rows.append({'dataset': ds_label, 'dataset_name': ds_name, 'budget': budget,
                         'macro_f1': None, 'accuracy': None, 'ece': None,
                         'inference_ms': None, 'params': None, 'peak_rss_MB': None, 'error': str(peak)[-300:]})
            continue
        f1, acc, ecev, lat, nparams = res
        print(f'{ds_label} {ds_name}: f1={f1:.4f} acc={acc:.4f} ece={ecev:.4f} lat={lat:.4f}ms params={nparams} peak={peak/1048576:.1f}MB ({time.time()-t0:.0f}s)', flush=True)
        rows.append({'dataset': ds_label, 'dataset_name': ds_name, 'budget': budget,
                     'macro_f1': f1, 'accuracy': acc, 'ece': ecev,
                     'inference_ms': lat, 'params': nparams, 'params_M': nparams / 1e6,
                     'peak_rss_MB': round(peak / 1048576, 2), 'error': ''})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'table_46_cgda_fixed_test.csv', index=False, float_format='%.4f')
    print('saved ->', OUT / 'table_46_cgda_fixed_test.csv')
    print(d.to_string(index=False))


if __name__ == '__main__':
    main()
