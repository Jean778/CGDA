# -*- coding: utf-8 -*-
"""GSA Concentration cross-concentration validation with network.

Protocol: LOCO - train 2 concentrations, test the 3rd (3 directions).
Budget: 2% calibration from target concentration.
Methods: base, cal_sup, mixup_std, cg_sup, cg_shuffle
Seeds: [7, 42, 123, 2024, 2025]
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

from revision_core import (StandardScaler, check_disjoint, train_network,
                           network_output, classification_metrics)
from run_revision import load_gsa

METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_shuffle']
SEEDS = [7, 42, 123, 2024, 2025]
BUDGET = 0.02
EPOCHS = 50
CONCS = [50, 100, 200]


def run_task(task):
    target_conc = task['target_conc']
    seed = task['seed']
    out_dir = PROJECT / 'results/gsa_loco' / f'c{target_conc}_s{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / 'completion.csv').exists():
        return task['task_id'], 'cached', 0

    t0 = time.time()
    try:
        x, y, conc = load_gsa(str(PROJECT))
        classes = np.arange(6)  # 0-5

        # source = the other two concentrations
        source = np.flatnonzero(conc != target_conc)
        target = np.flatnonzero(conc == target_conc)

        # calibration: 2 samples per class in the target concentration (as revision_v2 b2, cal=12)
        rng = np.random.RandomState(seed)
        cal_ids = []
        for c in classes:
            ids = target[y[target] == c]
            n = min(2, len(ids) - 1)
            cal_ids.extend(rng.choice(ids, size=n, replace=False).tolist())
        cal = np.array(cal_ids)
        test = np.setdiff1d(target, cal)
        check_disjoint(source_train=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool := cal) + 360) / 128)

        rows = []
        for m in METHODS:
            t1 = time.time()
            model, details = train_network(xs, y[source], xc, y[cal], m, seed, classes, EPOCHS, steps)
            logits, probs, _ = network_output(model, xt, classes)
            met = classification_metrics(y[test], logits, probs, classes)
            rows.append({
                'target_conc': target_conc,
                'source_concs': '+'.join(str(c) for c in CONCS if c != target_conc),
                'budget': BUDGET,
                'seed': seed,
                'method': m,
                'macro_f1': met['macro_f1'],
                'accuracy': met['accuracy'],
                'n_source': len(source),
                'n_cal': len(cal),
                'n_test': len(test),
                'seconds': time.time() - t1,
            })

        pd.DataFrame(rows).to_csv(out_dir / 'metrics.csv', index=False)
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0

    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def main():
    tasks = [dict(target_conc=c, seed=s, task_id=f'c{c}_s{s}') for c in CONCS for s in SEEDS]
    print(f'PLAN tasks={len(tasks)}  (3 conc x 5 seed, {len(METHODS)} methods each)', flush=True)
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
