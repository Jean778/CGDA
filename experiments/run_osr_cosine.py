# -*- coding: utf-8 -*-
"""OSR experiment with cosine distance instead of Mahalanobis.

Output format matches previous panel_44_*.csv files.
Budget: 0.01 first.
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
from run_revision import load_drift, shuffled_pool

METHOD = 'cg_sup'
BUDGET = 0.01
EPOCHS = 50

UNKNOWN_PAIRS = ['1-2', '3-5', '4-6']
SEEDS = [7, 42, 123, 2024, 2025]
FOLDS = list(range(1, 11))


def cosine_distance(feats, feats_ref):
    """Cosine distance: 1 - cosine similarity."""
    # feats: (n_test, dim)
    # feats_ref: (n_ref, dim)
    # return: (n_test,) min cosine distance to any ref
    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    ref_norm = feats_ref / (np.linalg.norm(feats_ref, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(feats_norm, ref_norm.T)  # (n_test, n_ref)
    dist = 1 - sim
    return np.min(dist, axis=1)


def run_task(task):
    unknown_pair = task['unknown_pair']
    budget = task['budget']
    seed = task['seed']
    fold = task['fold']
    out_dir = PROJECT / 'results/osr_cosine/tasks' / task['task_id']
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / 'completion.csv').exists():
        return task['task_id'], 'cached', 0

    t0 = time.time()
    try:
        x, y, b = load_drift(str(PROJECT))
        classes = np.arange(1, 7)
        known_classes = [1, 2, 3]
        unknown_classes = [4, 5, 6]

        # source = batch 1
        source = np.flatnonzero(b == 1)
        # target test = batch 2
        target = np.flatnonzero(b == 2)

        pool, test = stratified_parts(target, y, [.2, .8], seed)
        pool = shuffled_pool(pool, seed)
        cal = nested_calibration(pool, target, y, budget)
        check_disjoint(source=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 360) / 128)

        # train cg_sup
        model, details = train_network(xs, y[source], xc, y[cal], METHOD, seed, classes, EPOCHS, steps)

        # extract features
        _, _, f_source = network_output(model, xs, classes)
        _, _, f_test = network_output(model, xt, classes)

        is_unknown_test = np.isin(y[test], unknown_classes).astype(int)

        # cosine distance to class centroid (not nearest-neighbour sample)
        # per class: compute centroid, then cosine distance of test samples to it
        # final distance = min cosine distance over known-class centroids
        d_cos = np.full(len(f_test), np.inf)
        for c in known_classes:
            mask = y[source] == c
            if mask.sum() == 0:
                continue
            mu_c = f_source[mask].mean(0)
            # cosine similarity
            cos_sim = np.dot(f_test, mu_c) / (np.linalg.norm(f_test, axis=1) * np.linalg.norm(mu_c) + 1e-8)
            d_cos_c = 1 - cos_sim
            d_cos = np.minimum(d_cos, d_cos_c)

        # compute AUROC
        n1 = np.sum(is_unknown_test == 1)
        n0 = np.sum(is_unknown_test == 0)
        ranks = np.argsort(np.argsort(d_cos)) + 1
        u1 = np.sum(ranks[is_unknown_test == 1]) - n1 * (n1 + 1) / 2
        auroc = u1 / (n1 * n0) if (n1 * n0) > 0 else np.nan

        # save results
        results = pd.DataFrame({
            'sample_id': np.arange(len(xt)),
            'distance_score': d_cos,
            'is_unknown': is_unknown_test,
            'y_true': y[test],
        })
        results.to_csv(out_dir / 'predictions.csv', index=False)

        pd.DataFrame([{
            'family': 'osr',
            'setting': unknown_pair,
            'budget': budget,
            'seed': seed,
            'fold': fold,
            'method': METHOD,
            'auroc': auroc,
            'n_test': len(xt),
            'n_known': n0,
            'n_unknown': n1,
        }]).to_csv(out_dir / 'metrics.csv', index=False)

        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)

        return task['task_id'], 'complete', time.time() - t0

    except Exception as e:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(e)}]).to_csv(out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks():
    tasks = []
    for up in UNKNOWN_PAIRS:
        for seed in SEEDS:
            for fold in FOLDS:
                tasks.append(dict(
                    unknown_pair=up,
                    budget=BUDGET,
                    seed=seed,
                    fold=fold,
                    task_id=f'osr_{up}_b{BUDGET}_s{seed}_f{fold}'
                ))
    return tasks


def main():
    tasks = build_tasks()
    print(f'PLAN tasks={len(tasks)}  (cosine distance, budget={BUDGET})', flush=True)
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=1) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            tid, st, el = f.result()
            if st == 'complete': done += 1
            elif st == 'failed': fail += 1
            if i % 10 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] done={done}, fail={fail}, elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'FINISHED complete={done} failed={fail} wall={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
