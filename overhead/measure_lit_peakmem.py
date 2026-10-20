# -*- coding: utf-8 -*-
"""Measure peak RSS memory for lightweight literature models.

Each (dataset, model) runs in its own subprocess (training + inference).
The parent polls the child's RSS via psutil during the run and records the
peak, so the number covers PyTorch CPU allocations, numpy and Python overhead.

Output: main_results/table_46_lit_peakmem.csv
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
MODEL_ORDER = ['Sniff ConvNet', 'Sniff ResNet', 'GasNet', 'LSTM', 'LeNet-5', 'Sniff Multinose']

CHILD = textwrap.dedent(r'''
    import os, sys, math
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k] = '6'
    sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
    sys.path.insert(0, str(REPO / 'experiments'))
sys.path.insert(0, str(REPO / 'overhead'))
    import numpy as np
    from revision_core import StandardScaler, stratified_parts, check_disjoint
    from run_revision import load_drift
    from run_cdsl_full import load_cdsl
    from measure_lit_light_metrics import BUILDERS, train_base
    dataset, name = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3])
    if dataset == 'A':
        x, y, b = load_drift(str(REPO))
        source = np.flatnonzero(b == 1); target = np.flatnonzero(b == 2)
        classes = np.arange(1, 7)
    elif dataset == 'B':
        x, y, source, target = load_cdsl('master_slave1')
        classes = np.arange(1, 7)
    else:
        from run_gsa_loco import load_gsa
        x, y, conc = load_gsa(str(REPO)); y = y + 1
        source = np.flatnonzero(conc != 50); target = np.flatnonzero(conc == 50)
        classes = np.arange(1, 7)
    pool, test = stratified_parts(target, y, [.2, .8], seed)
    check_disjoint(source=source, test=test)
    sc = StandardScaler().fit(x[source])
    xs = sc.transform(x[source]).astype(np.float32)
    xt = sc.transform(x[test]).astype(np.float32)
    steps = math.ceil((len(source) + len(pool) + 360) / 128)
    model = BUILDERS[name](len(classes))
    model = train_base(model, xs, y[source], seed, classes, steps)
    model.eval()
    import torch
    with torch.no_grad():
        _ = model(torch.from_numpy(np.asarray(xt, np.float32).reshape(len(xt), 1, -1)))
    print('DONE')
''')


def run_sub(dataset, name):
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, '-c', CHILD, dataset, name, str(SEED)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    peak = 0
    while True:
        try:
            rss = psutil.Process(proc.pid).memory_info().rss
            peak = max(peak, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if proc.poll() is not None:
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            break
        time.sleep(0.05)
    out, err = proc.communicate()
    if proc.returncode != 0:
        return None, err.decode(errors='ignore')[-400:]
    return peak, None


def main():
    rows = []
    for dataset in ['A', 'B', 'C']:
        for name in MODEL_ORDER:
            t0 = time.time()
            peak, err = run_sub(dataset, name)
            if peak is None:
                print(f'{dataset} {name}: FAILED {err}', flush=True)
                rows.append({'dataset': dataset, 'model': name, 'peak_rss_MB': None, 'error': err})
            else:
                mb = peak / 1024 / 1024
                print(f'{dataset} {name}: peak_rss={mb:.1f} MB ({time.time()-t0:.0f}s)', flush=True)
                rows.append({'dataset': dataset, 'model': name, 'peak_rss_MB': round(mb, 2), 'error': ''})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'table_46_lit_peakmem.csv', index=False, float_format='%.2f')
    print('saved ->', OUT / 'table_46_lit_peakmem.csv')
    print(d.to_string(index=False))


if __name__ == '__main__':
    main()
