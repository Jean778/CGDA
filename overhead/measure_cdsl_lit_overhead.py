# -*- coding: utf-8 -*-
"""Measure CDSL lightweight literature models (runnable subset).

CDSL features are only 6-dim; 1D CNN/LSTM models built for 128-252-dim inputs:
  Runnable: Sniff ConvNet, Sniff ResNet, LSTM, LeNet-5
  Not runnable on 6-dim input: GasNet (MaxPool reduces 6->3->1->0),
                               Sniff Multinose (n_groups=4 -> seg=1 -> squeeze bug)

Directions: master->slave1, master->slave2. base (source-only) training, seed=7, EPOCHS=50.
Per (direction x model) fresh subprocess; parent polls RSS peak via psutil.
Records macro_f1, accuracy, ece, inference_ms, params, FLOPs (thop), peak_rss_MB, training_s.

Output: main_results/table_cdsl_lit_overhead.csv
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
MODEL_ORDER = ['Sniff ConvNet', 'Sniff ResNet', 'LSTM', 'LeNet-5']
DIRECTIONS = ['master_slave1', 'master_slave2']
N_FEAT = 6
N_CLASSES = 6

CHILD = textwrap.dedent(r'''
    import os, sys, math, time
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    sys.path.insert(0, os.path.join('__REPO__', 'core'))
    sys.path.insert(0, os.path.join('__REPO__', 'experiments'))
    sys.path.insert(0, os.path.join('__REPO__', 'overhead'))
    import numpy as np
    import pandas as pd
    import torch
    import thop
    from revision_core import StandardScaler, stratified_parts, check_disjoint, classification_metrics, ece
    from measure_lit_light_metrics import BUILDERS, train_base

    direction = sys.argv[1]
    name = sys.argv[2]
    seed = int(sys.argv[3])
    EPOCHS = int(sys.argv[4])

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
    check_disjoint(source=source, test=test)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)

    model = BUILDERS[name](len(classes))
    t0 = time.time()
    model = train_base(model, xs, y[source], seed, classes, steps)
    train_s = time.time() - t0
    model.eval()
    xx = torch.from_numpy(xt.reshape(len(xt), 1, -1))
    with torch.no_grad():
        _ = model(xx[:16])
        t1 = time.perf_counter(); _ = model(xx); t2 = time.perf_counter()
    lat = (t2 - t1) * 1000.0 / len(xt)
    with torch.no_grad():
        probs = torch.softmax(model(xx), 1).numpy()
    pred = np.asarray(classes)[probs.argmax(1)]
    m = classification_metrics(y[test], pred, probs, classes)
    ecev = ece(probs, y[test], classes)
    macs, params = thop.profile(model, inputs=(torch.zeros(1, 1, 6),), verbose=False)
    flops = macs * 2
    n_params = sum(p.numel() for p in model.parameters())
    print('RESULT|' + '|'.join([name, str(m['macro_f1']), str(m['accuracy']), str(ecev),
                                str(lat), str(n_params), str(flops), str(train_s)]))
    raise SystemExit(0)
''')


def run_sub(direction, name):
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, '-c', CHILD.replace('__REPO__', str(REPO)), direction, name, str(SEED), str(EPOCHS)],
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
        return None, err.decode(errors='ignore')[-500:]
    for line in out.decode(errors='ignore').splitlines():
        if line.startswith('RESULT|'):
            parts = line.strip().split('|')
            return (parts[1], float(parts[2]), float(parts[3]), float(parts[4]),
                    float(parts[5]), int(parts[6]), float(parts[7]), float(parts[8])), peak
    return None, 'no RESULT line'


def main():
    rows = []
    for direction in DIRECTIONS:
        for name in MODEL_ORDER:
            t0 = time.time()
            res, peak = run_sub(direction, name)
            if res is None:
                print(f'{direction}/{name}: FAILED {str(peak)[-300:]}', flush=True)
                rows.append({'direction': direction, 'model': name, 'method': 'base',
                             'macro_f1': None, 'accuracy': None, 'ece': None,
                             'inference_ms': None, 'params': None, 'params_M': None,
                             'flops': None, 'flops_M': None, 'peak_rss_MB': None,
                             'training_s': None, 'error': str(peak)[-300:]})
                continue
            meth, f1, acc, ecev, lat, nparams, flops, trs = res
            rows.append({'direction': direction, 'model': meth, 'method': 'base',
                         'macro_f1': f1, 'accuracy': acc, 'ece': ecev,
                         'inference_ms': lat, 'params': nparams, 'params_M': nparams / 1e6,
                         'flops': flops, 'flops_M': flops / 1e6,
                         'peak_rss_MB': round(peak / 1048576, 2),
                         'training_s': round(trs, 3), 'error': ''})
            print(f'{direction}/{meth}: f1={f1:.4f} acc={acc:.4f} ece={ecev:.4f} '
                  f'lat={lat:.4f}ms params={nparams} flops_M={flops/1e6:.4f} '
                  f'peak={peak/1048576:.1f}MB train={trs:.1f}s ({time.time()-t0:.0f}s)', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'table_cdsl_lit_overhead.csv', index=False, float_format='%.6f')
    print('saved ->', OUT / 'table_cdsl_lit_overhead.csv')
    print(d.to_string(index=False))


if __name__ == '__main__':
    main()
