# -*- coding: utf-8 -*-
"""Measure CDSL dataset overhead for all 6 methods (same caliber as table_46_cgda_fixed_test).

Directions: master->slave1 (CDSL_S1), master->slave2 (CDSL_S2)
budget = 0.10, seed = 7, EPOCHS = 50.

Per (direction x method) runs in one fresh subprocess; parent polls RSS peak via psutil.
Outputs macro_f1, accuracy, ece, inference_ms, params, macs, peak_rss_MB, training_s.

Output: main_results/table_cdsl_overhead.csv
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, math, time, subprocess, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

PROJECT = REPO
OUT = PROJECT / 'main_results'
SEED = 7
EPOCHS = 50
BUDGET = 0.10
METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_mix', 'cg_shuffle']
DIRECTIONS = ['master_slave1', 'master_slave2']

CHILD = textwrap.dedent(r'''
    import os, sys, math, time
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    sys.path.insert(0, os.path.join('__REPO__', 'core'))
    sys.path.insert(0, os.path.join('__REPO__', 'experiments'))
    import numpy as np
    import pandas as pd
    import torch
    from revision_core import (StandardScaler, stratified_parts, nested_calibration,
                               check_disjoint, train_network, network_output,
                               classification_metrics, ece)
    from run_revision import shuffled_pool

    direction = sys.argv[1]
    method = sys.argv[2]
    budget = float(sys.argv[3])
    seed = int(sys.argv[4])
    EPOCHS = int(sys.argv[5])

    DATA_DIR = os.path.join('__REPO__', 'data', 'CDSL_csv')
    fname = 'cdsl_master_slave1.csv' if direction == 'master_slave1' else 'cdsl_master_slave2.csv'
    df = pd.read_csv(DATA_DIR + '/' + fname)
    x = df[[f'f{i}' for i in range(1, 7)]].values.astype(np.float64)
    y = df['label'].values.astype(int)
    dom = df['domain'].values
    source = np.flatnonzero(dom == 'master')
    target = np.flatnonzero(dom != 'master')
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

    t0 = time.time()
    model, details = train_network(xs, y[source], xc, y[cal], method, seed, classes, EPOCHS, steps)
    train_s = time.time() - t0
    logits, probs, features = network_output(model, xt, classes)
    m = classification_metrics(y[test], logits, probs, classes)
    ecev = ece(probs, y[test], classes)

    # latency (batch-amortized, same as measure_cgda_fixed_test)
    model.eval()
    xxt = torch.from_numpy(xt.reshape(len(xt), 1, -1))
    with torch.no_grad():
        for _ in range(5):
            _ = model(xxt[:16])
        t1 = time.perf_counter()
        _ = model(xxt)
        t2 = time.perf_counter()
    lat = (t2 - t1) * 1000.0 / len(xt)

    n_params = details.get('parameter_count', sum(p.numel() for p in model.parameters()))
    # MACs: same counting as measure_overhead.count_macs (linear layers only)
    macs = 0
    prev = xs.shape[1]
    for layer in model.fe:
        if isinstance(layer, torch.nn.Linear):
            macs += prev * layer.out_features
            prev = layer.out_features
    macs += prev * model.cls.out_features

    print('RESULT', method, m['macro_f1'], m['accuracy'], ecev, lat, n_params, macs, train_s)
    raise SystemExit(0)
''')


def run_sub(direction, method):
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, '-c', CHILD.replace('__REPO__', str(REPO)), direction, method,
                             str(BUDGET), str(SEED), str(EPOCHS)],
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
            parts = line.split()
            return (parts[1], float(parts[2]), float(parts[3]), float(parts[4]),
                    float(parts[5]), int(parts[6]), int(parts[7]), float(parts[8])), peak
    return None, 'no RESULT line'


def main():
    rows = []
    for direction in DIRECTIONS:
        for method in METHODS:
            t0 = time.time()
            res, peak = run_sub(direction, method)
            if res is None:
                print(f'{direction}/{method}: FAILED {str(peak)[-300:]}', flush=True)
                rows.append({'direction': direction, 'method': method, 'budget': BUDGET,
                             'macro_f1': None, 'accuracy': None, 'ece': None,
                             'inference_ms': None, 'params': None, 'macs': None,
                             'peak_rss_MB': None, 'training_s': None, 'error': str(peak)[-300:]})
                continue
            meth, f1, acc, ecev, lat, nparams, nmacs, trs = res
            rows.append({'direction': direction, 'method': meth, 'budget': BUDGET,
                         'macro_f1': f1, 'accuracy': acc, 'ece': ecev,
                         'inference_ms': lat, 'params': nparams, 'params_M': nparams / 1e6,
                         'macs': nmacs, 'macs_M': nmacs / 1e6,
                         'peak_rss_MB': round(peak / 1048576, 2),
                         'training_s': round(trs, 3), 'error': ''})
            print(f'{direction}/{meth}: f1={f1:.4f} acc={acc:.4f} ece={ecev:.4f} '
                  f'lat={lat:.4f}ms params={nparams} macs={nmacs} '
                  f'peak={peak/1048576:.1f}MB train={trs:.1f}s ({time.time()-t0:.0f}s)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'table_cdsl_overhead.csv', index=False, float_format='%.4f')
    print('saved ->', OUT / 'table_cdsl_overhead.csv')
    print(d.to_string(index=False))


if __name__ == '__main__':
    main()
